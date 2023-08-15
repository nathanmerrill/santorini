#[macro_use]
extern crate lazy_static;
use serde::{Deserialize, Serialize};
use minimax::{Strategy, Game, ParallelSearch};

use colored::Colorize;
use rand::{rngs::ThreadRng, RngCore, Rng, seq::IteratorRandom};


fn main() {
    let mut population = Population::read();
    loop {
        println!("Iteration {}", population.iteration);
        population.diff_evolution();
        population.iteration += 1;
        population.save();
        
        //println!("Testing latest specimen");

        //run_full_game(population.population[0].clone());
    }
}


#[derive(Serialize, Deserialize)]
struct Population {
    iteration: usize,
    population: Vec<Eval>
}

const EVO_RATE: f64 = 0.5;
const CROSSOVER_RATE: f64 = 0.9;
const POP_SIZE: usize = 1000;
const FACTOR_COUNT: usize = 74;
const FILE_PATH: &str = "./src/weights.json";

impl Population {
    fn read() -> Self {
        if std::fs::metadata(FILE_PATH).is_ok() {
            Self::from_file()
        } else {
            Self::new_random()
        }
    }

    fn new_random() -> Self {
        Self {
            iteration: 0,
            population: (0..POP_SIZE).map(|_| Eval::random()).collect()
        }
    }

    fn save(&self) 
    {
        let value = serde_json::to_string_pretty(self).expect("Unable to serialize!");
        let mut file = OpenOptions::new().create(true).write(true).truncate(true).open(FILE_PATH).expect("Unable to open file!");
        file.write(value.as_bytes()).expect("Unable to write file!");
    }

    fn from_file() -> Self {
        let data = std::fs::read_to_string(FILE_PATH).expect("Unable to read file!");
        serde_json::from_str(&data).expect("Unable to parse file!")
    }

    fn diff_evolution(&mut self) 
    {
        let mut rng = rand::thread_rng();
        let mut next_population = vec![];
        for i in 0..self.population.len() {
            let choices = self.population.iter().choose_multiple(&mut rng, 3);
            
            let mutant = choices[2].mutate(choices[0], choices[1]);
            let trial = self.population[i].cross_over(&mutant);

            let peers = self.population.iter().choose_multiple(&mut rng, 10);

            let mut trial_count: usize = 0;
            let mut existing_count: usize = 0;
            for peer in peers {
                if compare_evals(peer.clone(), trial.clone()) {
                    trial_count += 1;
                }
                if !compare_evals(trial.clone(), peer.clone()) {
                    trial_count += 1;
                }
                if compare_evals(peer.clone(), self.population[i].clone()) {
                    existing_count += 1;
                }
                if !compare_evals(self.population[i].clone(), peer.clone()) {
                    existing_count += 1;
                }

            }
            if trial_count > existing_count {
                let diff: f64 = self.population[i].weights.iter().zip(&trial.weights).map(|(a, b)| (a - b).abs()).sum();
                next_population.push(trial);

                println!("Trial specimen won {} battles, baseline won {} battles. Replacing specimen {}. Weight Diff: {}", trial_count, existing_count, i, diff);
            } else {
                next_population.push(self.population[i].clone());
                println!("Trial specimen won {} battles, baseline won {} battles. Keeping specimen {}", trial_count, existing_count, i);
            }
        }

        self.population = next_population;
    }
}

fn run_full_game(eval: Eval) 
{
    let mut strategy = ParallelSearch::new(eval, minimax::IterativeOptions::default().with_mtdf().verbose(), minimax::ParallelOptions::new());    
    strategy.set_max_depth(6);
    let mut game = GameState::default();

    while !game.game_over {
        println!("{}", game);
        let _move = strategy.choose_move(&game).expect("No moves returned from strategy!");
        game = Santorini::apply(&mut game, _move).expect("State was not applied!");
    }
    
    println!("{}", game);

    if game.winner == Player::P1 {
        println!("Player 1 wins!");
    } else {
        println!("Player 2 wins!");
    }
}

fn compare_evals(baseline_eval: Eval, trial_eval: Eval) -> bool // Returns true if new eval is better than the old one
{
    let mut baseline = ParallelSearch::new(baseline_eval, minimax::IterativeOptions::default(), minimax::ParallelOptions::new());
    let mut new = ParallelSearch::new(trial_eval, minimax::IterativeOptions::default(), minimax::ParallelOptions::new());

    baseline.set_max_depth(4);
    new.set_max_depth(4);

    let baseline_player = Player::P1;
    
    let mut game = GameState::default();
    
    while !game.game_over {

        let eval = if game.current_player == baseline_player {
            &mut baseline
        } else {
            &mut new
        };
        
        let _move = eval.choose_move(&mut game).expect("No moves returned from strategy!");

        game = Santorini::apply(&mut game, _move).expect("State was not applied!");
    }

    return game.winner != baseline_player
}

// To run the search we need an evaluator.
#[derive(Clone, Serialize, Deserialize)]
struct Eval {
    weights: Vec<f64>
}

impl Eval {
    fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            weights: (0..FACTOR_COUNT).map(|_| rng.gen_range(-1.0..1.0)).collect()
        }
    }

    fn mutate(&self, cand_a: &Self, cand_b: &Self) -> Self {
        let weights: Vec<_> = (0..FACTOR_COUNT).map(|i| (self.weights[i] + EVO_RATE * (cand_b.weights[i] - cand_a.weights[i]))).collect();

        let max: f64 = weights.iter().map(|a| a.abs()).fold(0.0, f64::max);
        
        Self {
            weights: weights.into_iter().map(|a| a/max).collect()
        }
    }

    fn cross_over(&self, mutant: &Self) -> Self {
        let mut rng = rand::thread_rng();
        let position = rng.gen_range(0..74);
        Self {
            weights: (0..FACTOR_COUNT).map(|i| if i == position || rng.gen_bool(CROSSOVER_RATE) {mutant.weights[i]} else {self.weights[i]}).collect()
        }
    }
}


impl minimax::Evaluator for Eval {
    type G = Santorini;
    fn evaluate(&self, state: &GameState) -> minimax::Evaluation {
        if state.game_over {
            if state.winner == state.current_player {
                minimax::BEST_EVAL
            } else {
                minimax::WORST_EVAL
            }
        } else {
            let my_workers = state.worker_positions(state.current_player);
            let my_worker_heights = [state.squares[my_workers[0]].height as usize, state.squares[my_workers[1]].height as usize];
            let opp_workers = state.worker_positions(state.current_player.opposite());
            let opp_worker_heights = [state.squares[opp_workers[0]].height as usize, state.squares[opp_workers[1]].height as usize];
            
            let (my_high_worker, my_high_worker_height, my_low_worker, my_low_worker_height) = if my_worker_heights[0] < my_worker_heights[1] {(my_workers[1], my_worker_heights[1], my_workers[0], my_worker_heights[0])} else {(my_workers[0], my_worker_heights[0], my_workers[1], my_worker_heights[1])};
            let (opp_high_worker, opp_high_worker_height, opp_low_worker, opp_low_worker_height) = if opp_worker_heights[0] < opp_worker_heights[1] {(opp_workers[1], opp_worker_heights[1], opp_workers[0], opp_worker_heights[0])} else {(opp_workers[0], opp_worker_heights[0], opp_workers[1], opp_worker_heights[1])};

            let my_adjacent_high_heights = adjacent_heights(state, my_low_worker);
            let my_adjacent_low_heights = adjacent_heights(state, my_high_worker);
            let opp_adjacent_high_heights = adjacent_heights(state, opp_low_worker);
            let opp_adjacent_low_heights = adjacent_heights(state, opp_high_worker);

            let my_high_mobility = my_adjacent_high_heights[0] + my_adjacent_high_heights[1] + my_adjacent_high_heights[2] + my_adjacent_high_heights[3] + my_adjacent_high_heights[4];
            let my_low_mobility = my_adjacent_low_heights[0] + my_adjacent_low_heights[1] + my_adjacent_low_heights[2] + my_adjacent_low_heights[3] + my_adjacent_low_heights[4];
            let opp_high_mobility = opp_adjacent_high_heights[0] + opp_adjacent_high_heights[1] + opp_adjacent_high_heights[2] + opp_adjacent_high_heights[3] + opp_adjacent_high_heights[4];
            let opp_low_mobility = opp_adjacent_low_heights[0] + opp_adjacent_low_heights[1] + opp_adjacent_low_heights[2] + opp_adjacent_low_heights[3] + opp_adjacent_low_heights[4];

            let factors = [
                distance_between(my_high_worker, my_low_worker),
                distance_between(opp_high_worker, opp_low_worker),
                distance_between(my_low_worker, opp_high_worker),
                distance_between(my_low_worker, opp_low_worker),
                distance_between(my_high_worker, opp_high_worker),
                distance_between(my_high_worker, opp_low_worker),
                my_high_worker_height,
                my_low_worker_height,
                opp_high_worker_height,
                opp_low_worker_height,
                my_adjacent_high_heights[0],
                my_adjacent_high_heights[1],
                my_adjacent_high_heights[2],
                my_adjacent_high_heights[3],
                my_adjacent_high_heights[4], // Treat 1 level higher special as workers easily move up
                (my_adjacent_high_heights[4] == 0) as usize,
                (my_adjacent_high_heights[4] == 1) as usize,
                (my_adjacent_high_heights[4] == 2) as usize,
                my_adjacent_high_heights[5],
                my_adjacent_high_heights[6],
                my_adjacent_high_heights[7],
                my_adjacent_low_heights[0],
                my_adjacent_low_heights[1],
                my_adjacent_low_heights[2],
                my_adjacent_low_heights[3],
                my_adjacent_low_heights[4], // Treat 1 level higher special as workers easily move up
                (my_adjacent_low_heights[4] == 0) as usize,
                (my_adjacent_low_heights[4] == 1) as usize,
                (my_adjacent_low_heights[4] == 2) as usize,
                my_adjacent_low_heights[5],
                my_adjacent_low_heights[6],
                my_adjacent_low_heights[7],
                opp_adjacent_high_heights[0],
                opp_adjacent_high_heights[1],
                opp_adjacent_high_heights[2],
                opp_adjacent_high_heights[3],
                opp_adjacent_high_heights[4], // Treat 1 level higher special as workers easily move up
                (opp_adjacent_high_heights[4] == 0) as usize,
                (opp_adjacent_high_heights[4] == 1) as usize,
                (opp_adjacent_high_heights[4] == 2) as usize,
                opp_adjacent_high_heights[5],
                opp_adjacent_high_heights[6],
                opp_adjacent_high_heights[7],
                opp_adjacent_low_heights[0],
                opp_adjacent_low_heights[1],
                opp_adjacent_low_heights[2],
                opp_adjacent_low_heights[3],
                opp_adjacent_low_heights[4], // Treat 1 level higher special as workers easily move up
                (opp_adjacent_low_heights[4] == 0) as usize,
                (opp_adjacent_low_heights[4] == 1) as usize,
                (opp_adjacent_low_heights[4] == 2) as usize,
                opp_adjacent_low_heights[5],
                opp_adjacent_low_heights[6],
                opp_adjacent_low_heights[7],
                my_high_mobility,
                (my_high_mobility == 0) as usize,
                (my_high_mobility == 1) as usize,
                (my_high_mobility == 2) as usize,
                (my_high_mobility == 3) as usize,
                my_low_mobility,
                (my_low_mobility == 0) as usize,
                (my_low_mobility == 1) as usize,
                (my_low_mobility == 2) as usize,
                (my_low_mobility == 3) as usize,
                opp_high_mobility,
                (opp_high_mobility == 0) as usize,
                (opp_high_mobility == 1) as usize,
                (opp_high_mobility == 2) as usize,
                (opp_high_mobility == 3) as usize,
                opp_low_mobility,
                (opp_low_mobility == 0) as usize,
                (opp_low_mobility == 1) as usize,
                (opp_low_mobility == 2) as usize,
                (opp_low_mobility == 3) as usize,
            ];

            let sum: f64 = factors.into_iter().zip(&self.weights).map(|(factor, weight)| factor as f64 * weight).sum();
            
            return (sum*100.0).round() as i16
        }
    }
}

fn adjacent_heights(state: &GameState, square: usize) -> [usize; 8]
{
    let mut heights = [0; 8]; // Height offsets: -3, -2, -1, 0, +1, +2, +3, Dome
    let height = state.squares[square].height;
    for &adjacency in ADJACENCIES[square] {
        let square = state.squares[adjacency];
        if square.dome {
            heights[7] += 1;
        } else {
            heights[(square.height + 3 - height) as usize] += 1; // Add 3 to make subtraction never go negative
        }
    }

    heights
}


fn distance_between(square1: usize, square2: usize) -> usize {
    if square1 == square2 {
        0
    } else {
        let (y1, x1) = (square1 / 5, square1 % 5);
        let (y2, x2) = (square2 / 5, square2 % 5);

        let xdiff = if x1 < x2 {x2 - x1} else {x1 - x2};
        let ydiff = if y1 < y2 {y2 - y1} else {y1 - y2};

        xdiff.max(ydiff)
    }
}

use std::{fmt::Display, fs::OpenOptions, io::Write};

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

impl minimax::Game for Santorini 
{
    type M = GameMove;
    type S = GameState;

    fn notation(_state: &Self::S, _move: Self::M) -> Option<String> {
        Some("".to_string())
    }

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) 
    {
        if state.game_over
        {
            return;
        }
        if state.in_setup {
            state.place_workers(moves)
        } else {
            state.get_player_moves(moves);

            if moves.len() == 0 {
                moves.push(OUT_OF_MOVES)
            }
        }
    }

    fn zobrist_hash(_state: &Self::S) -> u64 {
        HASH_TABLE.hash(_state)
    }

    fn apply(s: &mut Self::S, _move: Self::M) -> Option<Self::S> 
    {
        let mut state = *s;
        if _move == OUT_OF_MOVES {
            state.current_player = state.current_player.opposite();
            state.game_over = true;
            state.winner = state.current_player;
        } else if state.in_setup {
            let worker1_pos = _move.step as usize;
            let worker2_pos = _move.build as usize;
            state.squares[worker1_pos].worker = true;
            state.squares[worker2_pos].worker = true;
            let offset = if state.current_player == Player::P1 {0} else {2};
            state.worker_positions[offset] = worker1_pos;
            state.worker_positions[offset+1] = worker2_pos;
            state.current_player = state.current_player.opposite();
            if state.current_player == Player::P1 {
                state.in_setup = false;
            }
        } else {
            let origin = _move.origin as usize;
            let destination = _move.step as usize;
            
            state.squares[destination].worker = state.squares[origin].worker;
            state.squares[origin].worker = false;
            for position in state.worker_positions.iter_mut() {
                if *position == origin {
                    *position = destination;
                }
            }
            
            if state.squares[destination].height == 3 && state.squares[origin].height == 2 {
                state.game_over = true;
                state.winner = state.current_player;
            } else {
                let build = _move.build as usize;
                if state.squares[build].height == 3 {
                    state.squares[build].dome = true;
                } else {
                    state.squares[build].height += 1;
                }
            }

            state.current_player = state.current_player.opposite();
        }

        Some(state)
    }

    fn get_winner(state: &Self::S) -> Option<minimax::Winner> 
    {
        if state.game_over {
            Some(if state.winner == state.current_player {
                minimax::Winner::PlayerToMove
            } else {
                minimax::Winner::PlayerJustMoved
            })
        } else {
            None
        }
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
struct Square {
    height: u8,
    dome: bool,
    worker: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameState {
    current_player: Player,
    winner: Player,
    game_over: bool,
    in_setup: bool,
    squares: [Square; 25],
    worker_positions: [usize; 4],
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameMove {
    origin: u8,
    step: u8,
    build: u8,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            current_player: Player::P1,
            winner: Player::P1,
            game_over: false,
            in_setup: true,
            squares: [Square {
                height: 0,
                dome: false,
                worker: false
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
                let worker_str = match square.worker {
                    false => " . ",
                    true => {
                        if self.worker_positions[0] == position || self.worker_positions[1] == position {
                            " 1 "
                        } else {
                            " 2 "
                        }                        
                    }
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

const OUT_OF_MOVES: GameMove = GameMove {origin: u8::MAX, step: u8::MAX, build: u8::MAX};

impl GameState {    
    fn worker_positions(self, player: Player) -> [usize; 2] 
    {
        match player {
            Player::P1 => [self.worker_positions[0], self.worker_positions[1]],
            Player::P2 => [self.worker_positions[2], self.worker_positions[3]],
        }
    }

    fn get_player_moves(self, moves: &mut Vec<GameMove>)
    {
        for position in self.worker_positions(self.current_player) {
            let position_u8 = position as u8;
            for &destination in ADJACENCIES[position] {
                let new_square = self.squares[destination];
                let destination_u8 = destination as u8;
                if !new_square.dome && !new_square.worker && new_square.height <= self.squares[position].height + 1 {
                    moves.push(GameMove {origin: position_u8, step: destination_u8, build: position_u8});
                    for &build in ADJACENCIES[destination] {
                        let build_square = self.squares[build];
                        if !build_square.dome && !build_square.worker
                        {
                            moves.push(GameMove {origin: position as u8, step: destination_u8, build: build as u8})
                        }
                    }
                }
            }
        }
    }

    fn place_workers(self, options: &mut Vec<GameMove>)
    {
        for i in 0..25 {
            for j in i+1 .. 25 {
                if !self.squares[i].worker && !self.squares[j].worker {
                    options.push(GameMove { origin: 0, step: i as u8, build: j as u8 });
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
        let mut hash = if state.current_player == Player::P1 { self.player_set[0] } else {self.player_set[1]};
        hash ^= match (state.game_over, state.winner) {
            (false, _) => self.winner_set[0],
            (true, Player::P1) => self.winner_set[1],
            (true, Player::P2) => self.winner_set[2]
        };

        for (index, square) in state.squares.into_iter().enumerate() {
            hash ^= self.height_set[index*3 + square.height as usize];
            if square.dome {
                hash ^= self.dome_set[index];
            }
        }
        for (index, worker) in state.worker_positions.into_iter().enumerate() {
            let offset = if index < 2 {0} else {25};
            hash ^= self.worker_set[offset + worker]
        }

        hash        
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

