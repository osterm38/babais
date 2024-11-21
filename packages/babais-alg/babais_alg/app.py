"""
TODO:
- game: contains many levels, and current locations
- engine: engine for handling events???
- level: contains current board (grid), and history, and current rules, and engine for generating next state
- grid: m x n array, containing pieces
- piece: contains object properties like image, is it icon or word image, current location?

rules:
- baba is you
- flag is win
- text is float

- given user has entered direction/stay-put
  - find all locations of pieces
  - figure out all existing rules
  - add all potential changes to queue based on user input and rules imposing moves
  - determine what changes will happen
  - update the board based on these changes
  
- eg: 
  - suppose layout of level is currently:
    w w w w w w
    w   f F   w W
    w   B I Y w I
    w   b N   w S
    w w w w w w
  - found existing locations:
    - (words are upper case, icons are lower case, same letter)
    - B: baba (word) (x-1, y)
    - I: is (x, y)
    - Y: you (x+1, y)
    - F: flag (x, y+1)
    - N: win (x, y-1)
    - W: wall (x+3, y+1)
    - I: is (x+3, y) 
    - S: stop (x+3, y-1)
    - b: baba (icon) (x-1, y-1)
    - f: flag (x-1, y+1)
    - w: wall [(x+-2, y+-2)] (surround words in rectangle):
  - found existing rules:
    - (word is push)
    - baba is you
    - flag is win
    - wall is stop
  - potential changes:
    - i = 0
    - curr_pieces_and_direction (initially all 'you's) = [(baba (icon), user input)]
    - while True:
      - next_pieces_and_direction = []
      - for piece, direction in curr_pieces_and_direction:
        - apply 'adjust' to piece in direction (like a push/move in that direction)
        - next_pieces_and_direction.extend(all affected found)
      - if len(next_pieces_and_direction) == 0:
        - break
  - alt alg:
    - pieces_and_directions = ... # [(baba (icon), up)]
    - for piece, dir in pieces_and_directions:
      - def can_piece_move_from_here(piece):
      - def can_piece_move_to_here(piece)
      - def check_adjust_direction(piece, dir):
        # check if piece can be adjusted from where it is to one square in that direction
        # otherwise fall back to staying where it is if it can't move
        - assert dir is not None
        # can it move based on rules governing it
        - explicit_rules_affecting_piece = explicit_rules_affecting(piece) # == ['baba is you']
        - rules_allow_adjust = (
            # TODO: update
                f'{piece_text} is stop' not in rules_affecting_piece
                and (f'{piece_text} is push' in rules_affecting_piece
                    or f'{piece_text} is you' in rules_affecting_piece')
            )
        - if not rules_allow_adjust: # short circuit when rules disallow movement
          - return None
        # can the next set of impacted pieces move if this were to move
        - next_pieces = get_adjacent_pieces(piece, dir)
        - next_pieces_hinder_adjust = all(piece_can_be_adjusted(p, dir) for p in next_pieces)
        - return next_pieces_hinder_adjust
given a piece and a direction, a piece can move to the next square
if all pieces in that square are ok with the move
"""

def create():
    return ''
