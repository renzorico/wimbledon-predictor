// data.js — Wimbledon 2026 Men's Singles draw data
// Last updated: 2026-07-02 (Day 3, R1 in progress)
// Update this file as results come in

const TOURNAMENT = {
  year: 2026,
  currentRound: 'R1',
  currentDay: 3,
  startDate: '2026-06-29',
  endDate: '2026-07-12',
  surface: 'Grass',
  defending: 'J. Sinner',
  notable: 'C. Alcaraz withdrew (wrist). J. Draper withdrew (arm).',
};

// Seed placement by quarter.
// Q1: 1,5,9,13,17,21,25,29  |  Q2: 3,7,11,15,19,23,27,31
// Q3: 4,8,12,16,20,24,28,32  |  Q4: 2,6,10,14,18,22,26,30

const SEEDS = [
  // -- Q1 --
  { seed: 1,  name: 'Jannik Sinner',              nation: 'ITA', quarter: 'Q1',
    r1: { opp: 'M. Kecmanovic (SRB)', result: 'W', score: '4-6 6-3 6-7(8) 6-2 6-3' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 5,  name: 'Alex de Minaur',             nation: 'AUS', quarter: 'Q1',
    r1: { opp: 'R. Burruchaga (ARG)', result: 'W', score: '7-6(5) 6-1 6-0' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 9,  name: 'Flavio Cobolli',             nation: 'ITA', quarter: 'Q1',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 13, name: 'Jiri Lehecka',               nation: 'CZE', quarter: 'Q1',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 17, name: 'Frances Tiafoe',             nation: 'USA', quarter: 'Q1',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 21, name: 'Tommy Paul',                 nation: 'USA', quarter: 'Q1',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 25, name: 'Arthur Rinderknech',         nation: 'FRA', quarter: 'Q1',
    r1: { opp: 'O. Tarvet (GBR)', result: 'W', score: '7-6(4) 7-6(4) 4-6 7-5' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 29, name: 'T.M. Etcheverry',            nation: 'ARG', quarter: 'Q1',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  // -- Q2 --
  { seed: 3,  name: 'Felix Auger-Aliassime',      nation: 'CAN', quarter: 'Q2',
    r1: { opp: 'A. Shevchenko (KAZ)', result: 'W', score: '6-3 6-1 6-4' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 7,  name: 'Novak Djokovic',             nation: 'SRB', quarter: 'Q2',
    r1: { opp: 'Y. Wu (CHN)', result: 'W', score: '6-4 5-7 6-4 6-4' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 11, name: 'Casper Ruud',                nation: 'NOR', quarter: 'Q2',
    r1: { opp: 'H. Hurkacz (POL)', result: 'L', score: '4-6 2-6 6-7(7)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 15, name: 'Jakub Mensik',               nation: 'CZE', quarter: 'Q2',
    r1: { opp: 'T. Samuel (GBR)', result: 'W', score: '7-5 3-6 3-6 6-3 7-6(7)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 19, name: 'Karen Khachanov',            nation: 'RUS', quarter: 'Q2',
    r1: { opp: 'B. Harris (GBR)', result: 'W', score: '6-3 7-5 6-3 6-3' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 23, name: 'Rafael Jodar',               nation: 'ESP', quarter: 'Q2',
    r1: { opp: 'F. Gill (GBR)', result: 'W', score: '6-3 6-3 7-5' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 27, name: 'Ugo Humbert',                nation: 'FRA', quarter: 'Q2',
    r1: { opp: 'Z. Bergs (BEL)', result: 'L', score: '2-6 5-7 6-4 6-3 3-6' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 31, name: 'Ignacio Buse',               nation: 'PER', quarter: 'Q2',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  // -- Q3 --
  { seed: 4,  name: 'Ben Shelton',                nation: 'USA', quarter: 'Q3',
    r1: { opp: 'O. Virtanen (FIN)', result: 'L', score: '6-4 3-6 6-7(8) 2-6 6-7(9)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 8,  name: 'Daniil Medvedev',            nation: 'RUS', quarter: 'Q3',
    r1: { opp: 'M. Cilic (CRO)', result: 'W', score: '6-1 6-2 6-4' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 12, name: 'Andrey Rublev',              nation: 'RUS', quarter: 'Q3',
    r1: { opp: 'R. Safiullin (RUS)', result: 'L', score: '4-6 7-6(6) 3-6 6-3 6-7(12)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 16, name: 'Learner Tien',               nation: 'USA', quarter: 'Q3',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 20, name: 'Arthur Fils',                nation: 'FRA', quarter: 'Q3',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 24, name: 'Joao Fonseca',               nation: 'BRA', quarter: 'Q3',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 28, name: 'Brandon Nakashima',          nation: 'USA', quarter: 'Q3',
    r1: { opp: 'J. Pinnington Jones (GBR)', result: 'W', score: '6-3 7-6(5) ...' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 32, name: 'Matteo Arnaldi',             nation: 'ITA', quarter: 'Q3',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  // -- Q4 --
  { seed: 2,  name: 'Alexander Zverev',           nation: 'GER', quarter: 'Q4',
    r1: { opp: 'A. Blockx (BEL)', result: 'W', score: '6-4 6-7(8) 7-6(5) 7-6(0)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 6,  name: 'Taylor Fritz',               nation: 'USA', quarter: 'Q4',
    r1: { opp: 'D. Lajovic (SRB)', result: 'W', score: '6-3 6-4 6-3' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 10, name: 'Alexander Bublik',           nation: 'KAZ', quarter: 'Q4',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 14, name: 'Luciano Darderi',            nation: 'ITA', quarter: 'Q4',
    r1: { opp: 'E. Quinn (USA)', result: 'L', score: '6-7(7) 5-7 2-6' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 18, name: 'Francisco Cerundolo',        nation: 'ARG', quarter: 'Q4',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 22, name: 'A. Davidovich Fokina',       nation: 'ESP', quarter: 'Q4',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 26, name: 'Cameron Norrie',             nation: 'GBR', quarter: 'Q4',
    r1: { opp: 'M. Zheng (USA)', result: 'L', score: '7-6(7) 2-6 7-6(2) 3-6 6-7(4)' },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },

  { seed: 30, name: 'Alejandro Tabilo',           nation: 'CHI', quarter: 'Q4',
    r1: { opp: null, result: null, score: null },
    r2: null, r3: null, r4: null, qf: null, sf: null, f: null },
];

// R1 upsets (seeded player lost)
const UPSETS = [
  { round: 'R1', winner: 'Otto Virtanen',   winNation: 'FIN', loserSeed: 4,  loser: 'B. Shelton',   loseNation: 'USA', score: '6-4 3-6 6-7(8) 6-2 7-6(9)' },
  { round: 'R1', winner: 'H. Hurkacz',      winNation: 'POL', loserSeed: 11, loser: 'C. Ruud',      loseNation: 'NOR', score: '6-4 6-2 7-6(7)' },
  { round: 'R1', winner: 'R. Safiullin',    winNation: 'RUS', loserSeed: 12, loser: 'A. Rublev',    loseNation: 'RUS', score: '6-4 6-7(6) 3-6 6-3 7-6(12)' },
  { round: 'R1', winner: 'E. Quinn',        winNation: 'USA', loserSeed: 14, loser: 'L. Darderi',   loseNation: 'ITA', score: '7-6(7) 7-5 6-2' },
  { round: 'R1', winner: 'M. Zheng',        winNation: 'USA', loserSeed: 26, loser: 'C. Norrie',    loseNation: 'GBR', score: '6-7(7) 6-2 6-7(2) 6-3 7-6(4)' },
  { round: 'R1', winner: 'Z. Bergs',        winNation: 'BEL', loserSeed: 27, loser: 'U. Humbert',   loseNation: 'FRA', score: '6-2 7-5 4-6 3-6 6-3' },
];

// Scoring weights
const SCORING = {
  R1: 1, R2: 2, R3: 4, R4: 8, QF: 16, SF: 32, F: 64,
  confidenceMultiplier: { H: 1.5, M: 1.0, L: 0.5 },
};

const QUARTER_LABELS = {
  Q1: 'Q1 — Sinner half',
  Q2: 'Q2 — Auger-Aliassime half',
  Q3: 'Q3 — Shelton / Medvedev half',
  Q4: 'Q4 — Zverev half',
};
