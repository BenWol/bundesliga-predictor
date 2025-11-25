"""
Kicktipp.de integration for submitting predictions.

Uses requests + BeautifulSoup to login and submit predictions
without requiring a browser.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from .config import KICKTIPP_EMAIL, KICKTIPP_PASSWORD, KICKTIPP_COMMUNITY


@dataclass
class KicktippMatch:
    """Represents a match on Kicktipp."""
    match_id: str
    home_team: str
    away_team: str
    home_field: str  # Form field name for home score
    away_field: str  # Form field name for away score
    current_home_tip: Optional[int] = None
    current_away_tip: Optional[int] = None


class KicktippClient:
    """
    Client for interacting with Kicktipp.de.

    Usage:
        client = KicktippClient()
        client.login()
        matches = client.get_current_matchday()
        client.submit_predictions({match_id: (home, away), ...})
    """

    BASE_URL = "https://www.kicktipp.de"

    def __init__(
        self,
        email: Optional[str] = None,
        password: Optional[str] = None,
        community: Optional[str] = None
    ):
        """
        Initialize Kicktipp client.

        Args:
            email: Kicktipp email (or from env KICKTIPP_EMAIL)
            password: Kicktipp password (or from env KICKTIPP_PASSWORD)
            community: Community name (or from env KICKTIPP_COMMUNITY)
        """
        self.email = email or KICKTIPP_EMAIL
        self.password = password or KICKTIPP_PASSWORD
        self.community = community or KICKTIPP_COMMUNITY

        if not self.email or not self.password:
            raise ValueError(
                "Kicktipp credentials required. Set KICKTIPP_EMAIL and "
                "KICKTIPP_PASSWORD in .env file or pass to constructor."
            )

        if not self.community:
            raise ValueError(
                "Kicktipp community required. Set KICKTIPP_COMMUNITY in "
                ".env file (e.g., 'wolter') or pass to constructor."
            )

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/120.0.0.0 Safari/537.36'
        })
        self._logged_in = False

    def login(self) -> bool:
        """
        Login to Kicktipp.

        Returns:
            True if login successful

        Raises:
            RuntimeError: If login fails
        """
        # First get the login page to establish session cookies
        login_page_url = f"{self.BASE_URL}/{self.community}/profil/login"
        resp = self.session.get(login_page_url)
        resp.raise_for_status()

        # Submit login form to the loginaction endpoint
        login_action_url = f"{self.BASE_URL}/{self.community}/profil/loginaction"
        login_data = {
            'kennung': self.email,
            'passwort': self.password,
            '_charset_': 'UTF-8',
        }

        resp = self.session.post(login_action_url, data=login_data, allow_redirects=True)
        resp.raise_for_status()

        # Check if login succeeded by looking for logout link or user menu
        if 'profil/logout' in resp.text or 'meinetipprunden' in resp.text:
            self._logged_in = True
            return True

        # Check for error message
        soup = BeautifulSoup(resp.text, 'html.parser')
        error = soup.find('div', class_='alert-danger')
        if error:
            raise RuntimeError(f"Login failed: {error.get_text(strip=True)}")

        raise RuntimeError("Login failed: Unknown error")

    def get_current_matchday(self) -> List[KicktippMatch]:
        """
        Get matches for the current matchday.

        Returns:
            List of KicktippMatch objects
        """
        if not self._logged_in:
            self.login()

        url = f"{self.BASE_URL}/{self.community}/tippabgabe"
        resp = self.session.get(url)
        resp.raise_for_status()

        return self._parse_matches(resp.text)

    def get_matchday(self, matchday: int) -> List[KicktippMatch]:
        """
        Get matches for a specific matchday.

        Args:
            matchday: Matchday number (1-34)

        Returns:
            List of KicktippMatch objects
        """
        if not self._logged_in:
            self.login()

        url = f"{self.BASE_URL}/{self.community}/tippabgabe?spieltagIndex={matchday}"
        resp = self.session.get(url)
        resp.raise_for_status()

        return self._parse_matches(resp.text)

    def _parse_matches(self, html: str) -> List[KicktippMatch]:
        """Parse matches from the tippabgabe page."""
        soup = BeautifulSoup(html, 'html.parser')
        matches = []

        # Find all match rows by looking for rows containing heimTipp inputs
        # Current format: spieltippForms[ID].heimTipp / spieltippForms[ID].gastTipp
        for home_input in soup.find_all('input', {'name': re.compile(r'\.heimTipp$')}):
            row = home_input.find_parent('tr')
            if not row:
                continue

            away_input = row.find('input', {'name': re.compile(r'\.gastTipp$')})
            if not away_input:
                continue

            home_field = home_input.get('name', '')
            away_field = away_input.get('name', '')

            # Extract match ID from field name (e.g., "spieltippForms[1389997051].heimTipp" -> "1389997051")
            match_id_match = re.search(r'\[(\d+)\]', home_field)
            match_id = match_id_match.group(1) if match_id_match else home_field

            # Get team names from cells with class 'nw'
            # Structure: [time, home_team, away_team, tipp_input, odds]
            team_cells = row.find_all('td', class_='nw')
            home_team = ""
            away_team = ""

            # Filter out cells that contain time or odds (have additional classes)
            team_only_cells = [
                cell for cell in team_cells
                if 'kicktipp-time' not in cell.get('class', [])
                and 'quoten' not in cell.get('class', [])
            ]

            if len(team_only_cells) >= 2:
                home_team = team_only_cells[0].get_text(strip=True)
                away_team = team_only_cells[1].get_text(strip=True)

            # Get current tips if any
            current_home = home_input.get('value', '')
            current_away = away_input.get('value', '')

            matches.append(KicktippMatch(
                match_id=match_id,
                home_team=home_team,
                away_team=away_team,
                home_field=home_field,
                away_field=away_field,
                current_home_tip=int(current_home) if current_home.isdigit() else None,
                current_away_tip=int(current_away) if current_away.isdigit() else None,
            ))

        return matches

    def submit_predictions(
        self,
        predictions: Dict[str, Tuple[int, int]],
        overwrite: bool = True,
        dry_run: bool = False
    ) -> Dict[str, bool]:
        """
        Submit predictions to Kicktipp.

        Args:
            predictions: Dict mapping match_id to (home_score, away_score)
            overwrite: Whether to overwrite existing predictions
            dry_run: If True, don't actually submit

        Returns:
            Dict mapping match_id to success status
        """
        if not self._logged_in:
            self.login()

        # Get current matchday to get form structure
        url = f"{self.BASE_URL}/{self.community}/tippabgabe"
        resp = self.session.get(url)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, 'html.parser')
        form = soup.find('form')

        if not form:
            raise RuntimeError("Could not find prediction form")

        # Build form data
        form_data = {}

        # Get all existing form fields (hidden fields, etc.)
        for inp in form.find_all('input'):
            name = inp.get('name')
            if name:
                form_data[name] = inp.get('value', '')

        # Get current matches
        matches = self._parse_matches(resp.text)
        results = {}

        for match in matches:
            if match.match_id in predictions:
                home, away = predictions[match.match_id]

                # Check if already has tip and overwrite is False
                if not overwrite and match.current_home_tip is not None:
                    results[match.match_id] = False
                    continue

                form_data[match.home_field] = str(home)
                form_data[match.away_field] = str(away)
                results[match.match_id] = True

        if dry_run:
            return results

        # Find form action URL
        action = form.get('action', '/tippabgabe')
        if not action.startswith('http'):
            action = f"{self.BASE_URL}/{self.community}{action}" if not action.startswith('/') else f"{self.BASE_URL}{action}"

        # Submit form
        resp = self.session.post(action, data=form_data)
        resp.raise_for_status()

        # Check for success (no error message)
        if 'alert-danger' in resp.text:
            soup = BeautifulSoup(resp.text, 'html.parser')
            error = soup.find('div', class_='alert-danger')
            if error:
                raise RuntimeError(f"Submission failed: {error.get_text(strip=True)}")

        return results

    def submit_from_predictor_results(
        self,
        predictions: List[Dict],
        overwrite: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Submit predictions from BundesligaPredictor results.

        Args:
            predictions: List of prediction dicts from predictor
            overwrite: Whether to overwrite existing predictions
            dry_run: If True, don't actually submit

        Returns:
            Dict mapping "home vs away" to (success, message)
        """
        if not self._logged_in:
            self.login()

        # Get Kicktipp matches
        kt_matches = self.get_current_matchday()

        results = {}
        matched_predictions = {}

        for pred in predictions:
            home_team = pred['home_team']
            away_team = pred['away_team']
            ensemble = pred['ensemble']
            home_score = ensemble['home']
            away_score = ensemble['away']

            # Find matching Kicktipp match
            match = self._find_matching_match(kt_matches, home_team, away_team)

            if match:
                matched_predictions[match.match_id] = (home_score, away_score)
                key = f"{home_team} vs {away_team}"

                if not overwrite and match.current_home_tip is not None:
                    results[key] = (False, f"Already has tip: {match.current_home_tip}-{match.current_away_tip}")
                else:
                    results[key] = (True, f"Will submit: {home_score}-{away_score}")
            else:
                key = f"{home_team} vs {away_team}"
                results[key] = (False, "Match not found on Kicktipp")

        if not dry_run and matched_predictions:
            self.submit_predictions(matched_predictions, overwrite=overwrite, dry_run=False)

        return results

    def _find_matching_match(
        self,
        kt_matches: List[KicktippMatch],
        home_team: str,
        away_team: str
    ) -> Optional[KicktippMatch]:
        """Find a Kicktipp match that matches the given teams."""
        # Normalize team names for matching
        def normalize(name: str) -> str:
            name = name.lower()
            # Common substitutions
            replacements = {
                'fc bayern münchen': 'bayern',
                'bayern munich': 'bayern',
                'bayer 04 leverkusen': 'leverkusen',
                'bayer leverkusen': 'leverkusen',
                'borussia dortmund': 'dortmund',
                'bvb': 'dortmund',
                'rb leipzig': 'leipzig',
                'rasenballsport leipzig': 'leipzig',
                'vfb stuttgart': 'stuttgart',
                'eintracht frankfurt': 'frankfurt',
                'sc freiburg': 'freiburg',
                'tsg 1899 hoffenheim': 'hoffenheim',
                'tsg hoffenheim': 'hoffenheim',
                'vfl wolfsburg': 'wolfsburg',
                'borussia mönchengladbach': 'gladbach',
                "bor. m'gladbach": 'gladbach',
                'werder bremen': 'bremen',
                'sv werder bremen': 'bremen',
                '1. fc union berlin': 'union berlin',
                'union berlin': 'union berlin',
                'fc augsburg': 'augsburg',
                '1. fc heidenheim 1846': 'heidenheim',
                'fc heidenheim': 'heidenheim',
                '1. fsv mainz 05': 'mainz',
                'mainz 05': 'mainz',
                'vfl bochum 1848': 'bochum',
                'vfl bochum': 'bochum',
                'fc st. pauli 1910': 'st. pauli',
                'fc st. pauli': 'st. pauli',
                'holstein kiel': 'kiel',
            }
            for full, short in replacements.items():
                if full in name:
                    return short
            return name.strip()

        home_norm = normalize(home_team)
        away_norm = normalize(away_team)

        for match in kt_matches:
            kt_home = normalize(match.home_team)
            kt_away = normalize(match.away_team)

            # Check if teams match (either direction)
            if (home_norm in kt_home or kt_home in home_norm) and \
               (away_norm in kt_away or kt_away in away_norm):
                return match

        return None
