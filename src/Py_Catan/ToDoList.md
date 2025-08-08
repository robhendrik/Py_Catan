To do list:
- Remove 'preference' from player base class. This only belongs to the child class value based player
- Remove calculate value hand, real estate and earning power functions
- For value based player build a check for a full game (or more). Find winning actions from best_action and generate value from explore that action. Then execute action and check is value is realized. Also check generate options and values for options to make sure value is the same. Also check that for all options the best_action has highest value. Make sure we mimic a game and set the _board and _other_players in the game for the players involved.
- Explicitely check if the copy board - execute action - calculate value works
- Add warning to calculate value if player does not know board or other players
- If player knows board and other players update after every build action. To save time we can remove this from board with the right comments or we do double to play safe.
- add a warning for all deprecated functions


Design logic:
- A Board() always has players, but players do not always have a board.
- Basic assumption: For any child the value only will depend on the hands, streets, villages and towns for all 4 players (i.e., the content of board vector).No other info from board or other players is needed.
- An action is executed as:
    - From base player class: Check if player can execute the actions (enough resources, still buildings left)
    - From base player class: Check what options the player has (what edges, nodes, trades are available)
    - From base player class: Generate a list of available actions
    - On board level: Execute player action and update impact for other 
    - On base player level: Calculate and track length of longest street for this player and whether player owns longest street
    - On board level: Check what is the longest street and whether it exceeds minimum, set the flags for owning longest street for all players.
    - From base player class: update the build options after (another) player has executed and action
    - For child player: Evaluate the list of possible actions  and attach a value (or take random decision)
    - On board level: Create a virtual new board where an action can be executed. The players are base players with same streets, hand, etc
    - On child player: Create a copy
    - On child player: Copy position from base class instance (copy streets, longest street etc)
    - For child player class: Evaluate the value on a virtual new board, for trades enforce the trade (i.e. always execute)
    - On board level: When playing a game, after creation of a board the players have to be informed of the board and their fellow players
    - 
