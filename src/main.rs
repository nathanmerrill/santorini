#[macro_use]
extern crate lazy_static;

use minimax::Strategy;

use crate::sim::*;

mod sim;


fn main() {
    go();
}
fn go() {
    let mut state = sim::INITIAL_GAME_STATE;

    let mut strategy = minimax::IterativeSearch::new(Eval{}, minimax::IterativeOptions::default());

    loop {
        println!("{}", display(&state));

        match strategy.choose_move(&state) {
            None => {
                if state[GAME_STATUS_VICTORY] {
                    if state[GAME_STATUS_WINNER] {
                        println!("Player 2 wins!")
                    } else {
                        println!("Player 1 wins!")
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
    type G = sim::Santorini;
    fn evaluate(&self, state: &sim::GameState) -> minimax::Evaluation {
        if state[GAME_STATUS_VICTORY] {
            if state[GAME_STATUS_WINNER] == state[GAME_PLAYER] {
                minimax::BEST_EVAL
            } else {
                minimax::WORST_EVAL
            }
        } else {
            let worker_max = player_workers(state[GAME_PLAYER], state).map(|a| height(square(a, state))).max().unwrap_or(0) as i16;
            let opponent_max = player_workers(!state[GAME_PLAYER], state).map(|a| height(square(a, state))).max().unwrap_or(0) as i16;
            worker_max - opponent_max                
        }
    }
}