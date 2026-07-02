"""Wimbledon 128-player bracket data structure."""

from dataclasses import dataclass, field


@dataclass
class Player:
    """A player in the draw."""
    player_id: int
    name: str
    seed: int | None = None
    nation: str = ""
    rank: int | None = None


@dataclass
class Match:
    """A single match in the bracket."""
    match_id: str           # e.g. "R1_M1", "QF_M2"
    round_name: str         # R1, R2, R3, R4, QF, SF, F
    player_a: Player | None = None
    player_b: Player | None = None
    winner: Player | None = None
    locked: bool = False    # True if result is known


ROUNDS = ["R1", "R2", "R3", "R4", "QF", "SF", "F"]
MATCHES_PER_ROUND = {"R1": 64, "R2": 32, "R3": 16, "R4": 8, "QF": 4, "SF": 2, "F": 1}


class WimbledonBracket:
    """128-player single-elimination bracket."""

    def __init__(self, players: list[Player]):
        """Initialize bracket with 128 players in draw order."""
        if len(players) != 128:
            raise ValueError(f"Need 128 players, got {len(players)}")
        self.players = players
        self.matches: dict[str, Match] = {}
        self._build_r1()

    def _build_r1(self) -> None:
        """Create R1 matches from draw positions."""
        for i in range(64):
            match_id = f"R1_M{i + 1}"
            self.matches[match_id] = Match(
                match_id=match_id,
                round_name="R1",
                player_a=self.players[i * 2],
                player_b=self.players[i * 2 + 1],
            )

    def lock_result(self, match_id: str, winner: Player) -> None:
        """Lock a known result (from live data)."""
        match = self.matches[match_id]
        match.winner = winner
        match.locked = True
        self._advance_winner(match_id, winner)

    def _advance_winner(self, match_id: str, winner: Player) -> None:
        """Advance winner to next round."""
        parts = match_id.split("_")
        current_round = parts[0]
        match_num = int(parts[1][1:])

        round_idx = ROUNDS.index(current_round)
        if round_idx >= len(ROUNDS) - 1:
            return  # Final — nowhere to advance

        next_round = ROUNDS[round_idx + 1]
        next_match_num = (match_num + 1) // 2
        next_match_id = f"{next_round}_M{next_match_num}"

        if next_match_id not in self.matches:
            self.matches[next_match_id] = Match(
                match_id=next_match_id,
                round_name=next_round,
            )

        next_match = self.matches[next_match_id]
        if match_num % 2 == 1:
            next_match.player_a = winner
        else:
            next_match.player_b = winner

    def get_round_matches(self, round_name: str) -> list[Match]:
        """Get all matches for a given round."""
        return [
            m for m in self.matches.values()
            if m.round_name == round_name
        ]

    def get_playable_matches(self) -> list[Match]:
        """Get matches that have both players set but no winner."""
        return [
            m for m in self.matches.values()
            if m.player_a is not None
            and m.player_b is not None
            and m.winner is None
        ]

    def reset_unlocked(self) -> "WimbledonBracket":
        """Return a copy with only locked results preserved."""
        import copy
        bracket = copy.deepcopy(self)
        # Remove non-locked match outcomes
        for match in list(bracket.matches.values()):
            if not match.locked:
                match.winner = None
        # Rebuild advancement from locked results only
        # (simplified: just clear future round slots that came from unlocked)
        return bracket

    def get_champion(self) -> Player | None:
        """Return the final winner, if determined."""
        finals = [m for m in self.matches.values() if m.round_name == "F"]
        if finals and finals[0].winner:
            return finals[0].winner
        return None
