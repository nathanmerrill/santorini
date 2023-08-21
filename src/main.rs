#[macro_use]
extern crate lazy_static;
use serde::{Deserialize, Serialize};
use minimax::{Strategy, Game, ParallelSearch};

use colored::Colorize;
use rand::{rngs::ThreadRng, RngCore, Rng};


const FACTOR_COUNT: usize = 42;
const HIDDEN_LAYER_COUNT: usize = 2;
const LEARNING_RATE: f64 = 1.0;
const FILE_PATH: &str = "./src/weights.json";

fn main() 
{
    let args: Vec<_> = std::env::args().collect();
    match args.get(1).map(|a| a.as_str()) {
        Some("Compare") => compare_history(),
        Some("Test") => test(),
        Some("Play") => play().unwrap(),
        Some(a) => panic!("Unexpected argument! {}", a),
        None => create_history()
    }
}

fn play() -> Result<(), std::io::Error>{
    println!("Welcome to Santorini!  Would you like to be player 1, 2, or Random?");
    let mut input = String::new();
    let player = loop {
       std::io::stdin().read_line(&mut input)?;
    
        match input.as_str().trim() {
            "1" => break Some(Player::P1),
            "2" => break Some(Player::P2),
            "R" | "Random" => break None,
            _ => println!("Invalid response. Please enter in '1', '2', or 'R'")
        }        
    };

    let player = player.unwrap_or_else(|| if rand::thread_rng().gen_bool(0.5) {Player::P1} else {Player::P2} );
    let eval = History::read().latest;
    let mut engine = ParallelSearch::new(eval.clone(), minimax::IterativeOptions::default(), minimax::ParallelOptions::new());  
    engine.set_timeout(Duration::from_secs(1));

    let mut game = GameState::default();

    while !game.game_over {
        println!("{}", game);
        let _move = 
        if game.current_player == player {
            input.clear();
            let mut available_moves = vec![];
            Santorini::generate_moves(&game, &mut available_moves);
            if available_moves.first() == Some(&OUT_OF_MOVES) {
                println!("You have no available moves!");
                OUT_OF_MOVES
            } else {

                if game.in_setup {
                    println!("Please enter the locations of your starting workers (X1,Y2):");
                } else {
                    println!("Please enter your next move: (X1:Y2+Z3 or X1:Y2)");
                }
                    
                *loop {
                    std::io::stdin().read_line(&mut input)?;
                    
                    match available_moves.iter().find(|a| a.notate().as_str().eq(input.trim())) {
                        Some(a) => break a,
                        None => match game.in_setup {
                            true => println!("Invalid input. Please enter the coordinates of where you would like your two workers, starting with the male. \nThe coordinates should be separated by a comma.\nExample: B3,A1 means to place the male worker on the 2nd column, 3rd row, and the female worker on the first column and the first row"),
                            false => println!("Invalid input. Please enter the coordinates of where you would like to move then build. \nThe initial coordinate indicates which worker to move, the next coordinate indicates where to move to, and the last coordinate indicates where to build. \nFor example, B2:C2+C3 means to move the worker on B2 to C2, then build on C3")
                        }
                    }
                }
            }          
            
        } else {
            let _move = engine.choose_move(&game).expect("No moves returned!");
            println!("I choose: {}", _move.notate());
            _move
        };

        game = Santorini::apply(&mut game, _move).expect("State was not applied!");   
    }

    if game.winner == player {
        println!("You won!")
    } else {
        println!("I won!")
    }

    Ok(())
}

fn test() {
    let inputs = [[2.7810836,2.550537003],
    [1.465489372,2.362125076],
    [3.396561688,4.400293529],
    [1.38807019,1.850220317],
    [3.06407232,3.005305973],
    [7.627531214,2.759262235],
    [5.332441248,2.088626775],
    [6.922596716,1.77106367],
    [8.675418651,-0.242068655],
    [7.673756466,3.508563011]];
    let outputs = [0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0];

    let mut eval = Eval::new(2);

    for i in 0..20 {
        let mut error_sum = 0.0;
        for (input, expected) in inputs.iter().zip(outputs) {
            let (output, hidden_outputs) = eval.forward_propagate(input);
            error_sum += eval.back_propagate(expected, output, &hidden_outputs, input, LEARNING_RATE);
            
        }
        println!("Epoch: {} Error: {}", i, error_sum)
    }
    
}

fn create_history() {

    let mut history = History::read();
    history.save();
    loop {
        println!("Iteration {}", history.latest.iteration);
        train(&mut history.latest);
        history.latest.iteration += 1;
        
        if history.latest.iteration.is_power_of_two() {
            history.evals.push(history.latest.clone());
        }

        history.save();
    }
}

fn compare_history() {
    
    let history = History::read();
    for eval in history.evals
    {
        if compare(eval.clone(), history.latest.clone()) {
            println!("Iteration {} lost to final iteration {}", eval.iteration, history.latest.iteration);
        } else {
            println!("Iteration {} beat final iteration {}", eval.iteration, history.latest.iteration);
        }
    }
}

fn compare(baseline: Eval, target: Eval) -> bool
{
    let mut baseline = ParallelSearch::new(baseline, minimax::IterativeOptions::default(), minimax::ParallelOptions::new());
    baseline.set_max_depth(6);
    let mut target = ParallelSearch::new(target, minimax::IterativeOptions::default(), minimax::ParallelOptions::new());
    target.set_max_depth(6);

    let mut game = GameState::default();

    while !game.game_over {
        let engine = if game.current_player == Player::P1 {
            &mut baseline
        } else {
            &mut target
        };
        let best_move = engine.choose_move(&game).expect("No moves returned!");
        game = Santorini::apply(&mut game, best_move).expect("State was not applied!");   
    }

    game.winner != Player::P1
}

fn train(eval: &mut Eval)
{
    let mut strategy = ParallelSearch::new(eval.clone(), minimax::IterativeOptions::default(), minimax::ParallelOptions::new());    
    strategy.set_max_depth(4);
    let mut game = GameState::default();

    let mut history = vec![];

    while !game.game_over {
        let best_move = strategy.choose_move(&game).expect("No moves returned from strategy!");
        let factors = Eval::get_factors(&game);
        let evaluation = eval.forward_propagate(&factors).0;
        let mut future_state = game.clone();
        for _move in strategy.principal_variation() {
            future_state = Santorini::apply(&mut future_state, _move).expect("State was not applied!");
        }

        let future_eval = if future_state.game_over {
            if future_state.winner == game.current_player {
                1.0
            } else {
                0.0
            }
        } else {
            eval.forward_propagate(&Eval::get_factors(&future_state)).0
        };

        history.push((game.clone(), future_eval));

        println!("{}", game);
        println!("Factors: {:?}", factors);
        println!("Future eval: {}, Current eval: {}, Error: {}", future_eval, evaluation, (future_eval - evaluation).abs());

        game = Santorini::apply(&mut game, best_move).expect("State was not applied!");
    }


    for (i,(state, future_eval)) in history.into_iter().rev().enumerate()
    {
        let input = Eval::get_factors(&state);
        let (evaluation, hidden_layers) = eval.forward_propagate(&input);
        let rate = LEARNING_RATE*10.0 / (i+10) as f64;
        let error = eval.back_propagate(future_eval, evaluation, &hidden_layers, &input, rate);
        println!("Training at rate {:.3}.  Future eval: {:.3} Current eval: {:.3} Reported error: {}", rate, future_eval, evaluation, error);
    }
    
    println!("{}", game);

    if game.winner == Player::P1 {
        println!("Player 1 wins!");
    } else {
        println!("Player 2 wins!");
    }
}

#[derive(Clone, Serialize, Deserialize)]
struct History {
    latest: Eval,
    evals: Vec<Eval>
}

impl History {
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

    fn read() -> Self {
        if std::fs::metadata(FILE_PATH).is_ok() {
            Self::from_file()
        } else {
            Self::new()
        }
    }
    fn new() -> Self {
        let eval = Eval::new(FACTOR_COUNT);
        Self {
            latest: eval.clone(),
            evals: vec![eval]
        }
    }
}

// To run the search we need an evaluator.
#[derive(Clone, Serialize, Deserialize)]
struct Eval {
    iteration: usize,
    factor_count: usize,
    hidden_layer: Vec<f64>, // Collapsed 2d array.  Each (FACTOR_COUNT+1) elements is a separate node
    output_node: Vec<f64>,
}

impl Eval {
    fn new(factor_count: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            iteration: 0,
            factor_count,
            hidden_layer: (0..(HIDDEN_LAYER_COUNT * (factor_count+1))).map(|_|rng.gen_range(0.0..1.0)).collect(),
            output_node: (0..HIDDEN_LAYER_COUNT+1).map(|_|rng.gen_range(0.0..1.0)).collect()
        }
    }

    fn activate(input: &[f64], layer: &[f64]) -> f64 {
        let mut iter = layer.iter();
        let bias = iter.next().unwrap();
        let sum: f64 = iter.zip(input).map(|(weight, input)| weight * input).sum();
        1.0 / ((-bias - sum).exp() + 1.0)
    }

    fn forward_propagate(&self, input: &[f64]) -> (f64, Vec<f64>) 
    {
        let hidden: Vec<f64> = self.hidden_layer.chunks_exact(self.factor_count+1).map(|a| {
            Self::activate(input, a)
        }).collect();
        

        let output = Self::activate(&hidden, &self.output_node);

        (output, hidden)
    }

    fn back_propagate(&mut self, expected: f64, output: f64, hidden_outputs: &Vec<f64>, input: &[f64], rate: f64) -> f64
    {
        let output_error = (output - expected) * output * (1.0-output);
        let hidden_errors: Vec<_> = self.output_node.iter()
            .skip(1)
            .zip(hidden_outputs).map(|(weight, hidden_output)| {
                weight * output_error * hidden_output * (1.0-hidden_output)
            }).collect();
        
        for (node, &hidden_error) in self.hidden_layer.chunks_exact_mut(self.factor_count+1).zip(&hidden_errors) {
            Self::train_node(node, hidden_error, input, rate);
        }
    
        Self::train_node(&mut self.output_node, output_error, hidden_outputs, rate);

        hidden_errors.into_iter().map(|a| a.abs()).sum::<f64>() + output_error.abs()
    }

    fn train_node(node: &mut [f64], error: f64, inputs: &[f64], rate: f64) 
    {
        let mut node_iter = node.iter_mut();
        *node_iter.next().unwrap() -= rate * error; // Bias node

        for (weight, input) in node_iter.zip(inputs) 
        {
            *weight -= rate * error * input
        }
    }

    fn get_factors(state: &GameState) -> Vec<f64> 
    {
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

        let mut factors = vec![
            distance_between(my_high_worker, my_low_worker),
            distance_between(opp_high_worker, opp_low_worker),
            distance_between(my_low_worker, opp_high_worker),
            distance_between(my_low_worker, opp_low_worker),
            distance_between(my_high_worker, opp_high_worker),
            distance_between(my_high_worker, opp_low_worker),
            my_high_worker_height as f64 / 3.0,
            my_low_worker_height as f64 / 3.0,
            opp_high_worker_height as f64 / 3.0,
            opp_low_worker_height as f64 / 3.0
        ];
        factors.extend(my_adjacent_high_heights.map(|a| a as f64 / 8.0));
        factors.extend(my_adjacent_low_heights.map(|a| a as f64 / 8.0));
        factors.extend(opp_adjacent_high_heights.map(|a| a as f64 / 8.0));
        factors.extend(opp_adjacent_low_heights.map(|a| a as f64 / 8.0));

        factors
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
            return (self.forward_propagate(&Self::get_factors(state)).0 * 1000.0).round() as i16;
        }
    }
}

fn adjacent_heights(state: &GameState, square: usize) -> [usize; 6]
{
    let mut heights = [0; 6];
    for &adjacency in ADJACENCIES[square] {
        let square = state.squares[adjacency];
        if square.dome {
            heights[5] += 1;
        } else if square.worker {
            heights[4] += 1;
        } else {
            heights[(square.height) as usize] += 1
        }
    }

    heights
}


fn distance_between(square1: usize, square2: usize) -> f64 {
    if square1 == square2 {
        0.0
    } else {
        let (y1, x1) = (square1 / 5, square1 % 5);
        let (y2, x2) = (square2 / 5, square2 % 5);

        let xdiff = if x1 < x2 {x2 - x1} else {x1 - x2};
        let ydiff = if y1 < y2 {y2 - y1} else {y1 - y2};

        xdiff.max(ydiff) as f64 / 5.0
    }
}

use std::{fmt::Display, fs::OpenOptions, io::Write, time::Duration};

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
        Some(_move.notate())
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

impl GameMove {
    fn notate(&self) -> String {
        match (self.origin, self.step, self.build) {
            (u8::MAX, u8::MAX, u8::MAX) => "S".to_owned(),
            (u8::MAX, a, b) => GameMove::to_coord(a).to_owned() + "," + GameMove::to_coord(b),
            (a, b, u8::MAX) => GameMove::to_coord(a).to_owned() + ":" + GameMove::to_coord(b),
            (a, b, c) => GameMove::to_coord(a).to_owned() + ":" + GameMove::to_coord(b) + "+"+GameMove::to_coord(c),        
        }
    }

    fn to_coord(index: u8) -> &'static str {
        match index {
            0 => "A1",
            1 => "B1",
            2 => "C1",
            3 => "D1",
            4 => "E1",
            5 => "A2",
            6 => "B2",
            7 => "C2",
            8 => "D2",
            9 => "E2",
            10 => "A3",
            11 => "B3",
            12 => "C3",
            13 => "D3",
            14 => "E3",
            15 => "A4",
            16 => "B4",
            17 => "C4",
            18 => "D4",
            19 => "E4",
            20 => "A5",
            21 => "B5",
            22 => "C5",
            23 => "D5",
            24 => "E5",
            _ => panic!("Unexpected coordinate!")
        }
    }
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
                if self.squares[position].height == 2 && self.squares[destination].height == 3 {
                    moves.push(GameMove { origin: position_u8, step: destination_u8, build: u8::MAX });
                    continue;
                }

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
            for j in 0 .. 25 {
                if !self.squares[i].worker && !self.squares[j].worker && i != j {
                    options.push(GameMove { origin: u8::MAX, step: i as u8, build: j as u8 });
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

