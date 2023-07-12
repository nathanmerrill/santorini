use bitvec::prelude::*;
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


// 1 bit for current turn
// 2 bits for game status
// 3 bits for height data for each square
// 5 bits for the location of each worker
pub type GameState = BitArr!(for (1+2+25*3+5*4));

pub const GAME_PLAYER: usize = 0;
pub const GAME_STATUS_VICTORY: usize = 1;
pub const GAME_STATUS_WINNER: usize = 2;
const GAME_STATUS_SETUP_FINISHED: usize = 2;
const GAME_STATE_MAP_OFFSET: usize = 3;
const GAME_STATE_WORKER_OFFSET: usize = 78;

const WORKER_REMOVED: usize = 25;

pub const INITIAL_GAME_STATE: GameState = BitArray::ZERO;

const P1: bool = false;
const P2: bool = true;


pub struct Santorini;

struct ZobristTable {
    player_set: [u64; 2],
    height_set: [u64; 100],
    dome_set: [u64; 25],
    worker_set: [u64; 50]
}

impl ZobristTable {
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            player_set: hash_table(&mut rng),
            height_set: hash_table(&mut rng),
            dome_set: hash_table(&mut rng),
            worker_set: hash_table(&mut rng),  
        }
    }

    pub fn hash(&self, state: &GameState) -> u64{
        let mut start = if state[GAME_PLAYER] { self.player_set[0] } else {self.player_set[1]};
        for (index, square) in squares(state).enumerate() {
            start ^= self.height_set[index*3 + (height(square) as usize)];
            if has_dome(square) {
                start ^= self.dome_set[index];
            }
        }

        for (index, worker) in workers(state).enumerate() {
            let offset = if index < 2 {0} else {25};
            start ^= self.worker_set[offset + worker]
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
        Some(display(_state))
    }

    fn generate_moves(state: &Self::S, moves: &mut Vec<Self::M>) 
    {
        *moves = get_next_moves(state);
        if moves.len() == 0 {
            println!("{}", display(state))
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
        if state[GAME_STATUS_VICTORY]
        {
            Some(if state[GAME_STATUS_WINNER] == state[GAME_PLAYER] {minimax::Winner::PlayerToMove} else {minimax::Winner::PlayerJustMoved})
        } else {
            None
        }
    }
}

fn squares(state: &GameState) -> impl Iterator<Item = &BitSlice>
{
    state[GAME_STATE_MAP_OFFSET..GAME_STATE_WORKER_OFFSET].chunks(3)
}

pub fn square(index: usize, state: &GameState) -> &BitSlice
{
    let offset = GAME_STATE_MAP_OFFSET + 3*index;
    &state[offset..offset+3]
}

fn has_dome(square: &BitSlice) -> bool
{
    square[0]
}

pub fn height(square: &BitSlice) -> u8 
{
    square[1..3].load()
}

fn set_height(position: usize, height: u8, state: &mut GameState) 
{
    let offset = GAME_STATE_MAP_OFFSET+position*3+1;
    state[offset..offset+2].store(height);
}

fn set_dome(position: usize, state: &mut GameState) 
{
    let offset = GAME_STATE_MAP_OFFSET+position*3;
    state.set(offset, true)
}

fn set_worker_position(worker_index: usize, position: usize, state: &mut GameState) 
{
    let offset = (worker_index * 5) + GAME_STATE_WORKER_OFFSET;
    state[offset..offset+5].store(position)
}

pub fn player_workers(player: bool, state: &GameState) -> impl Iterator<Item = usize> + '_
{
    let offset = (if player == P1 {0} else {10}) + GAME_STATE_WORKER_OFFSET;
    state[offset..offset+10].chunks(5).map(|a| a.load::<usize>()).filter(|&a| a != WORKER_REMOVED)
}

fn workers(state: &GameState) -> impl Iterator<Item = usize> + '_
{
    state[GAME_STATE_WORKER_OFFSET..GAME_STATE_WORKER_OFFSET+20].chunks(5).map(|a| a.load::<usize>()).filter(|&a| a != WORKER_REMOVED)
}

pub fn display(state: &GameState) -> String
{
    let mut display = String::from("|---------------");
    for (position, square) in squares(state).enumerate(){
        if position % 5 == 0 {
            display.push_str("|\n|");
        }

        if has_dome(square) {
            display.push_str("^^^");
        } else {
            let height = height(square).to_string().chars().next().unwrap();
            let mut worker_char = ' ';
            for (worker_index, worker) in workers(state).enumerate() {
                if worker == position {
                    if worker_index < 2 {
                        worker_char = 'X'
                    } else {
                        worker_char = 'Y'
                    }
                }
            }

            display.push(height);
            display.push(worker_char);
            display.push(height);
        }
    }
    
    display.push_str("|\n|---------------|");

    display
}


pub fn get_next_moves(state: &GameState) -> Vec<GameState> 
{
    if state[GAME_STATUS_VICTORY] {
        vec![*state]
    } else if !state[GAME_STATUS_SETUP_FINISHED] {
        place_initial_workers(state)
    } else {
        let next_moves = get_player_moves(state);

        if next_moves.len() == 0 {
            let mut new = *state;
            new.set(GAME_STATUS_VICTORY, true);
            new.set(GAME_STATUS_WINNER, !state[GAME_PLAYER]);
            new.set(GAME_PLAYER, !state[GAME_PLAYER]);

            vec![new]
        } else {
            next_moves
        }
    }
}

fn get_player_moves(state: &GameState) -> Vec<GameState>
{
    player_workers(state[GAME_PLAYER], state)
        .flat_map(|square| 
            get_worker_actions(state, square)
        ).map(|mut a| {
            toggle_player(&mut a);
            a
        }).collect()
}

fn get_worker_actions<'a>(state: &'a GameState, worker: usize) -> Vec<GameState>
{
    get_worker_moves(state, worker)
    .into_iter()
    .flat_map(|(worker, state)| 
        get_worker_builds(&state, worker)
    ).collect()
}

fn adjacencies(position: usize) -> Vec<usize> 
{
    match position {
        0 => vec![1,5,6],
        1|2|3 => vec![position-1, position+1, position+4, position+5, position+6],
        4 => vec![3,8,9],
        5|10|15 => vec![position-5, position-4, position+1, position+5, position+6],
        9|14|19 => vec![position-6, position-5, position-1, position+4, position+5],
        20 => vec![15,16,21],
        21|22|23 => vec![position-6, position-5, position-4, position-1, position+1],
        24 => vec![18,19,23],
        _ => vec![position-6, position-5, position-4, position-1, position+1, position+4, position+5, position+6]
    }
}

fn get_worker_moves<'a> (state: &'a GameState, worker: usize) -> Vec<(usize, GameState)>
{
    let current_height = height(square(worker, state));
    let mut moves = vec![];
    for destination in adjacencies(worker) 
    {
        let square = square(destination, state);
        let new_height = height(square);

        if !workers(state).any(|a| a == destination) && !has_dome(square) && new_height <= current_height + 1
        {
            let mut new_state = *state;
            move_worker(&mut new_state, worker, destination);
            if new_height == 3 && current_height == 2 {
                new_state.set(GAME_STATUS_VICTORY, true);
                let current_player = new_state[GAME_PLAYER];
                new_state.set(GAME_STATUS_WINNER, current_player);
            }

            moves.push((destination, new_state))
        }
    }

    moves
}

fn get_worker_builds<'a>(state: &'a GameState, worker: usize) -> Vec<GameState>
{
    if state[GAME_STATUS_VICTORY] 
    {
        return vec![*state]
    }

    let mut builds = vec![];
    for location in adjacencies(worker) 
    {
        let square = square(location, state);
        if !workers(state).any(|a| a == location) && !has_dome(square)
        {
            let mut new_state = *state;
            let square_height = height(square);
            if square_height == 3 {
                set_dome(location, &mut new_state);
            } else {
                set_height(location, square_height+1, &mut new_state);
            }

            builds.push(new_state)
        }
    }

    builds
}




fn toggle_player(state: &mut GameState) {
    let current_player = state[GAME_PLAYER];
    state.set(GAME_PLAYER, !current_player)
}

fn place_initial_workers(state: &GameState) -> Vec<GameState>
{
    let current_player = state[GAME_PLAYER];
    let worker_index = if current_player == P1 {0} else {2};

    place_initial_worker(state, worker_index)
    .iter()
    .flat_map(|s|
        place_initial_worker(s, worker_index+1)
    ).map(|mut a| {
        toggle_player(&mut a);
        if current_player == P2 {
            a.set(GAME_STATUS_SETUP_FINISHED, true)
        }
        a
    })
    .collect()
}

fn move_worker(state: &mut GameState, worker: usize, destination: usize)
{
    let worker_index = workers(state).position(|f| f == worker).unwrap();
    set_worker_position(worker_index, destination, state);
}

fn place_initial_worker(state: &GameState, index: usize) -> Vec<GameState>
{
    let mut options = vec![];
    for i in 0..25 {
        if !workers(state).take(index).any(|position| position == i) {
            let mut new: GameState = *state;
            set_worker_position(index, i, &mut new);
            options.push(new);
        }
    }

    options
}