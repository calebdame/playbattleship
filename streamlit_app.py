import streamlit as st
from battleship import BattleshipBoard, BattleshipPlayer

# To maintain the state between runs
boards_sim = 25000

if 'init' not in st.session_state:
    st.session_state.init = True
    st.session_state.guess_count = 0
    st.session_state.player = BattleshipPlayer(boards=boards_sim)
    st.session_state.enemy_board = [['~' for _ in range(10)] for _ in range(10)]

st.title("Battleship Game")

if st.session_state.init:
    st.session_state.init = False
    st.session_state.guess_count = 0
    st.session_state.player.generate_random_boards()
    st.session_state.enemy_board = [['~' for _ in range(10)] for _ in range(10)]

# Create widgets for user input
st.sidebar.text("Player's Turn")

# Display the best guess before the player makes a guess
best_guess = st.session_state.player.take_turn()
st.sidebar.text(f'Best guess: {best_guess}\n(Using {boards_sim} Simulations)')

if st.sidebar.button("Use Suggested Guess"):
    x_coordinate, y_coordinate = best_guess

    # Handle user input and update the game state
    st.session_state.player.update_game_state(x_coordinate, y_coordinate)
    
    # Update the board based on the guess result
    if (x_coordinate, y_coordinate) in st.session_state.player.hits:
        st.session_state.enemy_board[y_coordinate][x_coordinate] = 'H'
        st.session_state.feedback_message = 'Hit!'
    else:
        st.session_state.enemy_board[y_coordinate][x_coordinate] = 'M'
        st.session_state.feedback_message = 'Miss!'
    
    # Increase the guess counter
    st.session_state.guess_count += 1
    
    # Check if all ships are sunk
    if st.session_state.player.check_all_sunk():
        st.session_state.feedback_message = 'Congratulations, you won!'
    
    st.rerun()

x_coordinate = st.sidebar.slider('Choose your X coordinate:', 0, 9, 0)
y_coordinate = st.sidebar.slider('Choose your Y coordinate:', 0, 9, 0)

# Display the current state of the board
st.write("Enemy Board:")
st.table(st.session_state.enemy_board)

if st.sidebar.button("Submit Guess"):
    # Handle user input and update the game state
    st.session_state.player.update_game_state(x_coordinate, y_coordinate)
    
    # Update the board based on the guess result
    if (x_coordinate, y_coordinate) in st.session_state.player.hits:
        st.session_state.enemy_board[y_coordinate][x_coordinate] = 'H'
        st.session_state.feedback_message = 'Hit!'
    else:
        st.session_state.enemy_board[y_coordinate][x_coordinate] = 'M'
        st.session_state.feedback_message = 'Miss!'
    
    # Increase the guess counter
    st.session_state.guess_count += 1
    
    # Check if all ships are sunk
    if st.session_state.player.check_all_sunk():
        st.session_state.feedback_message = 'Congratulations, you won!'
    
    st.experimental_rerun()

# Display the feedback message if it exists
if hasattr(st.session_state, 'feedback_message'):
    st.sidebar.text(st.session_state.feedback_message)
    # Reset feedback message so it doesn't persist across multiple reruns
    del st.session_state.feedback_message

st.sidebar.text(f'Guess count: {st.session_state.guess_count}')

if st.sidebar.button("Reset Game"):
    st.session_state.init = True
