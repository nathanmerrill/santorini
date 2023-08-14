#[macro_use]
extern crate lazy_static;

use minimax::Strategy;

use colored::Colorize;


fn main() {
    go();
}
fn go() {
    let mut state = GameState::default();

    let mut strategy: minimax::Negamax<Eval> = minimax::Negamax::new(Eval{}, 6);
    //let mut strategy: MonteCarloTreeSearch<Santorini> = minimax::MonteCarloTreeSearch::new(minimax::MCTSOptions::default());
    //let mut strategy = minimax::IterativeSearch::new(Eval{}, minimax::IterativeOptions::default());

    loop {
        println!("{}", state);

        match strategy.choose_move(&state) {
            None => {
                if let Some(winner) = state.winner {
                    if winner == Player::P1 {
                        println!("Player 1 wins!")
                    } else {
                        println!("Player 2 wins!")
                    }
                } else {
                    println!("Unexpected game end!");
                }

                break;
            }
            Some(new_state) => {
                state = new_state
            }
        }
    }
}

// To run the search we need an evaluator.
struct Eval;
impl minimax::Evaluator for Eval {
    type G = Santorini;
    fn evaluate(&self, state: &GameState) -> minimax::Evaluation {
        if let Some(winner) = state.winner {
            if winner == state.current_player {
                minimax::BEST_EVAL
            } else {
                minimax::WORST_EVAL
            }
        } else { 
            let [p1w1, p1w2, p2w1, p2w2] = state.worker_positions.map(|a| state.squares[a].height);
            let p1 = p1w1*p1w1 + p1w2*p1w2;
            let p2 = p2w1*p2w1 + p2w2*p2w2;
            if state.current_player == Player::P1 {
                (p1 * 16) as i16 - (p2 * 8) as i16
            } else {
                (p2 * 16) as i16 - (p1 * 8) as i16
            }
            
        }
    }
}

use std::fmt::Display;

use rand::RngCore;
use rand::rngs::ThreadRng;

pub enum God {
    None,
    Apollo,
    Artemis,
    Athena,
    Atlas,
    Demeter,
    Hephaestus,
    Hermes,
    Minotaur,
    Pan,
    Prometheus,
    // Advanced
    Aphrodite,
    Ares,
    Bia,
    Chaos,
    Charon,
    Chronus,
    Circe,
    Dionysus,
    Eros,
    Hera,
    Hestia,
    Hypnus,
    Limus,
    Medusa,
    Morpheus,
    Persephone,
    Poseidon,
    Selene,
    Triton,
    Zeus,
    // Golden Fleece Expansion
    Aeolus,
    Charybdis,
    Clio,
    EuropaAndTalus,
    Gaea,
    Graeae,
    Hades,
    Harpies,
    Hecate,
    Moerae,
    Nemesis,
    Siren,
    Tartarus,
    Terpsichore,
    Urania,
    // Heroes
    Achilles,
    Adonis,
    Atalanta,
    Bellerophon,
    Heracles,
    Jason,
    Medea,
    Odysseus,
    Polyphemus,
    Theseus,
    // Promo Cards
    Asteria,
    CastorAndPollux,
    Hippolyta,
    Hydra,
    Scylla,
    Tyche
}

pub struct Santorini;

struct ZobristTable {
    player_set: [u64; 2],
    winner_set: [u64; 3],
    height_set: [u64; 100],
    dome_set: [u64; 25],
    worker_set: [u64; 50]
}

impl ZobristTable {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            player_set: hash_table(&mut rng),
            winner_set: hash_table(&mut rng),
            height_set: hash_table(&mut rng),
            dome_set: hash_table(&mut rng),
            worker_set: hash_table(&mut rng),  
        }
    }

    pub fn hash(&self, state: &GameState) -> u64{
        let mut start = if state.current_player == Player::P1 { self.player_set[0] } else {self.player_set[1]};
        start ^= match state.winner {
            None => self.winner_set[0],
            Some(Player::P1) => self.winner_set[1],
            Some(Player::P2) => self.winner_set[2]
        };

        for (index, square) in state.squares.into_iter().enumerate() {
            start ^= self.height_set[index*3 + square.height as usize];
            if square.dome {
                start ^= self.dome_set[index];
            }
            match square.worker {
                None => {},
                Some(worker) => {
                    let offset = match worker.player {
                        Player::P1 => 0,
                        Player::P2 => 25
                    };
                    start ^= self.worker_set[offset + index]
                }
            }
        }

        start        
    }
}

fn hash_table<const N: usize>(rng: &mut ThreadRng) -> [u64; N]
{
    std::array::from_fn(move |_|rng.next_u64())
}

lazy_static! {
    static ref HASH_TABLE: ZobristTable = {
        ZobristTable::new()
    };
}

impl minimax::Game for Santorini 
{
    type M = GameState;
    type S = GameState;

    fn notation(_state: &Self::S, _move: Self::M) -> Option<String> {
        Some("".to_string())
    }

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) 
    {
        state.get_next_moves(moves);
        if moves.len() == 0 {
            println!("{}", state)
        }
    }

    fn zobrist_hash(_state: &Self::S) -> u64 
    {
        HASH_TABLE.hash(_state)
    }

    fn apply(_: &mut Self::S, m: Self::M) -> Option<Self::S> 
    {
        Some(m)
    }

    fn get_winner(state: &Self::S) -> Option<minimax::Winner> 
    {
        state.winner.map(|a| {
            if a == state.current_player {
                minimax::Winner::PlayerToMove
            } else {
                minimax::Winner::PlayerJustMoved
            }
        })
    }
}


#[derive(Clone, Copy, PartialEq, Eq)]
enum Player {
    P1,
    P2
}

impl Player {
    fn opposite(self) -> Self {
        match self {
            Player::P1 => Player::P2,
            Player::P2 => Player::P1
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Worker {
    player: Player,
    female: bool,
}
impl Worker {
    fn index(self) -> usize {
        match (self.player, self.female) {
            (Player::P1, false) => 0,
            (Player::P1, true) => 1,
            (Player::P2, false) => 2,
            (Player::P2, true)  => 3
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Square {
    height: u8,
    dome: bool,
    worker: Option<Worker>,
}

impl Square {
    fn empty(self) -> bool {
        self.worker == None && !self.dome
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    current_player: Player,
    winner: Option<Player>,
    in_setup: bool,
    squares: [Square; 25],
    worker_positions: [usize; 4],
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            current_player: Player::P1,
            winner: None,
            in_setup: true,
            squares: [Square {
                height: 0,
                dome: false,
                worker: None
            }; 25],
            worker_positions: [0; 4],
        }
    }
}

impl Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("|---------------")?;
        for (position, square) in self.squares.iter().enumerate(){
            if position % 5 == 0 {
                write!(f, "|\n|")?;
            }

            if square.dome {
                " ^ ".bold().fmt(f)?;
            } else {
                let worker_str = match square.worker.map(|a| a.player) {
                    None => " . ",
                    Some(Player::P1) => " 1 ",
                    Some(Player::P2) => " 2 ",
                };

                let colored_str = match square.height {
                    0 => worker_str.on_purple(),
                    1 => worker_str.on_blue(),
                    2 => worker_str.on_green(),
                    3 => worker_str.on_yellow(),
                    _ => panic!("Unexpected height")
                };

                colored_str.fmt(f)?;
            }
        }
        
        write!(f, "|\n|---------------|")
    }
}

impl GameState {
    pub fn get_next_moves(self, states: &mut Vec<GameState>)
    {
        if self.in_setup {
            self.place_workers(states)
        } else if self.winner.is_none() {
            self.get_player_moves(states);

            if states.len() == 0 {
                let mut new = self;
                new.current_player = self.current_player.opposite();
                new.winner = Some(new.current_player);
                states.push(new);
            }
        }
    }
    
    fn worker_positions(self, player: Player) -> [usize; 2] 
    {
        match player {
            Player::P1 => [self.worker_positions[0], self.worker_positions[1]],
            Player::P2 => [self.worker_positions[2], self.worker_positions[3]],
        }
    }

    fn get_player_moves(self, moves: &mut Vec<GameState>)
    {
        for position in self.worker_positions(self.current_player) {
            self.get_worker_moves(position, moves)
        }
    }    

    fn get_worker_moves(self, position: usize, moves: &mut Vec<GameState>)
    {
        let square = self.squares[position];
        let worker: Worker = square.worker.expect("No worker at position!");
        let worker_index = worker.index();
        for &destination in ADJACENCIES[position]
        {
            let new_square = self.squares[destination];
            if new_square.empty() && new_square.height <= square.height + 1
            {
                let mut new_state = self;
                new_state.squares[destination].worker = Some(worker);
                new_state.squares[position].worker = None;
                new_state.worker_positions[worker_index] = destination;
                new_state.current_player = self.current_player.opposite();

                if new_square.height == 3 && square.height == 2 {
                    new_state.winner = Some(self.current_player);
                    moves.push(new_state);
                    continue;
                }

                new_state.get_worker_builds(destination, moves);
            }
        }
    }

        
    fn get_worker_builds(self, worker: usize, moves: &mut Vec<GameState>)
    {
        for &location in ADJACENCIES[worker] 
        {
            let square = self.squares[location];
            if square.empty()
            {
                let mut new_state = self;
                if square.height == 3 {
                    new_state.squares[location].dome = true;
                } else {
                    new_state.squares[location].height += 1;
                }

                moves.push(new_state)
            }
        }
    }
        
    fn place_workers(self, options: &mut Vec<GameState>)
    {
        for i in 0..25 {
            for j in i+1 .. 25 {
                if self.squares[i].empty() && self.squares[j].empty() {
                    let mut new = self;
                    let worker1 = Worker{player: self.current_player, female: false};
                    let worker2 = Worker{player: self.current_player, female: true};
                    new.squares[i].worker = Some(worker1);
                    new.squares[j].worker = Some(worker2);
                    new.worker_positions[worker1.index()] = i;
                    new.worker_positions[worker2.index()] = j;
                    new.current_player = new.current_player.opposite();
                    if new.current_player == Player::P1 {
                        new.in_setup = false;
                    }
                    options.push(new);
                }
            }
        }
    }
}


lazy_static! {
    static ref ADJACENCIES: [&'static [usize]; 25] = core::array::from_fn(adjacencies);
}


fn adjacencies(position: usize) -> &'static [usize]
{
    match position {
        0 => &[1,5,6],
        1 => &[0,2,5,6,7],
        2 => &[1,3,6,7,8],
        3 => &[2,4,7,8,9],
        4 => &[3,8,9],
        5 => &[0,1,6,10,11],
        6 => &[0,1,2,5,7,10,11,12],
        7 => &[1,2,3,6,8,11,12,13],
        8 => &[2,3,4,7,9,12,13,14],
        9 => &[3,4,8,13,14],
        10 => &[5,6,11,15,16],
        11 => &[5,6,7,10,12,15,16,17],
        12 => &[6,7,8,11,13,16,17,18],
        13 => &[7,8,9,12,14,17,18,19],
        14 => &[8,9,13,18,19],
        15 => &[10,11,16,20,21],
        16 => &[10,11,12,15,17,20,21,22],
        17 => &[11,12,13,16,18,21,22,23],
        18 => &[12,13,14,17,19,22,23,24],
        19 => &[13,14,18,23,24],
        20 => &[15,16,21],
        21 => &[15,16,17,20,22],
        22 => &[16,17,18,21,23],
        23 => &[17,18,19,22,24],
        24 => &[18,19,23],
        _ => panic!("Unexpected value in adjacencies function")
    }
}
