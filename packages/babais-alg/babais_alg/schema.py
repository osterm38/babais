"""
'baba is you' is a valid sentence or rule comprised of 3 consecutive text blocks in the 'world':
- 'baba' (subject): (noun) text block pointing to the image block of the same name
- 'is' (verb): (verb) text block pointing to the action to be performed
- 'you' (object): () is the object point to where my playable character

rules:
- 'baba is push': each IBaba object should be pushed out of the square it is in if an object moves onto it
- 'baba is baba': each IBaba object should stay as that object
- 'baba is wall': each IBaba object should be replaced with a IWall object
- 'baba is you':  each IBaba object should be controlled by the user
- 'baba is win':  each IBaba object should be the winning location
- 'baba makes wall': each IBaba object should produce a new IWall object in place
- 'baba is open': each IBaba object can remove a shut object, removing itself in the process

- rules:
  - subject verb object, where
    - subject:
      - noun [conjunction noun [...]]
    - object:
      - noun/adjective [conjunction noun/adjective [...]]
  - nouns can be negated
  - adjectives can be negated

block (sub)types:
- text
  - noun (baba, wall, robot, me, text, empty, ...)
  - verb (is, has, makes, ...)
  - adjective (push, you, pull, win, shift, move, defeat, tele, ...)
  - conjunction (and, on)
  - prefix (not)

- a block has a type, faces a direction
- a square has a set of blocks and a location on a grid
- a grid has contiguous squares in xy space
- a level has a current and sequence of historical grids
- a ramble is a sequence of arbitrary word blocks
- a rule represents a ramble that is a valid sentence, which can be/is broken into its subject, verb, and object
  - a subject is a compound phrase where conjunctions separate one or more (noun-only) phrases
  - an object is a compound phrase where conjunctions separate one or more (simple noun or adjective) phrases
  - e.g.: [(leaf) <and> (not baba) <and> (robot on water)] [is] [{push} <and> {not empty}]
    - verb = is
    - subject (simple/complex noun) phrases: leaf (simple noun), not baba (simple noun), robot on water (conditional noun)
    - object (simple noun/adjective) phrases: push (simple adjective), not empty (simple noun)
    

"""
from enum import IntEnum
from itertools import product, chain
from pydantic import BaseModel, Field, conlist
from typing import Self, Iterator


class Location(BaseModel):
    x: int
    y: int

    def __add__(self, other: "Location") -> "Location":
        o = other.as_loc() if isinstance(other, Direction) else other
        return Location(
            x=self.x + o.x,
            y=self.y + o.y,
        )


class Direction(IntEnum):
    # direction, tied with how it'd add to a location
    LEFT  = -2
    DOWN  = -1
    STAY  = 0
    UP    = 1
    RIGHT = 2
    
    def as_loc(self, rel_to: Location = Location(x=0, y=0)) -> Location:
        # return location relative to origin (0, 0)
        return rel_to + Location(
            x=(1 if self == self.RIGHT else (-1 if self == self.LEFT else 0)),
            y=(1 if self == self.UP else (-1 if self == self.DOWN else 0)),
        )

class Block(BaseModel):
    pass
    # facing: Direction = Direction.DOWN

# *** word blocks **********************************
# blocks that only ever appear as a single text word
# **************************************************

class WordBlock(Block):
    pass

class Conditional(WordBlock):
    pass

class On(Conditional):
    pass

class Facing(Conditional):
    pass

CONDITIONAL_CLASSES = (On, Facing)


class Conjunction(WordBlock):
    pass

class And(Conjunction):
    pass

CONJUNCTION_CLASSES = (And,)


class Prefix(WordBlock):
    pass

class Not(Prefix):
    pass

PREFIX_CLASSES = (Not,)


class Verb(WordBlock):
    pass

class Is(Verb):
    pass

class Has(Verb):
    pass

class Makes(Verb):
    pass

VERB_CLASSES = (Is, Has, Makes)


class Adjective(WordBlock):
    pass

class You(Adjective):
    pass

class Win(Adjective):
    pass

class Shift(Adjective):
    pass

class Stop(Adjective):
    pass

class Push(Adjective):
    pass

class Pull(Adjective):
    pass

class More(Adjective):
    pass

ADJECTIVE_CLASSES = (You, Win, Shift, Stop, Push, Pull, More)


class Noun(WordBlock):
    pass
    # def image_cls(self) -> type[Block]:
    #     raise NotImplementedError()

class Baba(Noun):
    pass

class Flag(Noun):
    pass

class Text(Noun):
    pass

class Me(Noun):
    pass

class Robot(Noun):
    pass

class Empty(Noun):
    pass

class All(Noun):
    pass

NOUN_CLASSES = (Baba, Flag, Text, Me, Robot, Empty, All)


# *** images blocks ********************
# blocks that only ever appear as images
# **************************************

class ImageBlock(Block):
    pass

class IBaba(ImageBlock):
    pass

class IFlag(ImageBlock):
    pass

class IMe(ImageBlock):
    pass

class IRobot(ImageBlock):
    pass

class IEmpty(ImageBlock):
    pass

IMAGE_CLASSES = (IBaba, IFlag, IMe, IRobot, IEmpty)


class Ramble(BaseModel):
    words: list[WordBlock]
    
    def __len__(self) -> int:
        return len(self.words)
    
    def __iter__(self) -> Iterator[WordBlock]:
        return iter(self.words)
    
    def split_iter(self, by: type[WordBlock] = Conjunction, include_split_word: bool = False) -> Iterator["Ramble"] | Iterator[tuple["Ramble", WordBlock | None]]:
        # baba and not robot and -> [[baba], [not, robot], []]
        # baba is you -> [[baba, is, you]]
        # -> [[]]
        lst: list[WordBlock] = []
        for word in self:
            if isinstance(word, by):
                yield (Ramble(words=lst), word) if include_split_word else Ramble(words=lst)
                lst = []
            else:
                lst.append(word)
        yield (Ramble(words=lst), None) if include_split_word else Ramble(words=lst)
    
    def split(self, by: type[WordBlock] = Conjunction, include_split_word: bool = False) -> list["Ramble"] | list[tuple["Ramble", WordBlock | None]]:
        return list(self.split_iter(by=by, include_split_word=include_split_word))

    def rules_iter(self) -> Iterator["Rule"]:
        """generate any/all rules found, from left to right"""
        verb_idxs = [i for i, word in enumerate(self) if isinstance(word, Verb)]
        start_idxs = [0] + [i+1 for i in verb_idxs]
        verb_idxs.append(len(self))
        end_idxs = verb_idxs[1:]

        for start_idx, verb_idx, end_idx in zip(start_idxs, verb_idxs, end_idxs):
            verb = self.words[verb_idx]
            # print(f'{left_idx=}, {verb_idx=}, {end_idx=}, {self.num_words=}')
            for left_idx, right_idx in product(range(start_idx, verb_idx), range(end_idx, verb_idx, -1)):
                left_words = self.words[left_idx:verb_idx]
                right_words = self.words[verb_idx+1:right_idx]
                # print(f'{left_idx=}, {left_words=}')
                # print(f'{right_idx=}, {right_words=}')
                subj = Ramble(words=left_words)
                obj = Ramble(words=right_words)
                if subj.is_subject() and obj.is_object():
                    yield Rule(subject=subj, verb=verb, object=obj)
                    # only one rule can be formed between verbs
                    break
    
    def to_rules(self) -> list["Rule"]:
        return list(self.rules_iter())
    
    def to_rule(self, idx: int = 0):# -> "Rule" | None:
        for i, rule in enumerate(self.rules_iter()):
            if i == idx:
                return rule
        return None
    
    def is_verb(self):
        return len(self) == 1 and isinstance(self.words[0], Verb)
    
    def is_noun_phrase(self) -> bool:
        conds = self.split(by=Conditional)
        return (len(conds) <= 2) and all(cond.is_simple(_type=Noun) for cond in conds)

    def is_subject(self) -> bool:
        # subject means: noun [conj noun [conj noun...]]
        return all(phrase.is_noun_phrase() for phrase in self.split(by=Conjunction))
        
    def is_object(self):
        # object means: noun|adj [conj noun|adj [conj noun|adj...]]
        return all(phrase.is_simple(_type=Noun) or phrase.is_simple(_type=Adjective) for phrase in self.split(by=Conjunction))
    
    def is_simple(self, _type: type[WordBlock]) -> bool:
        return not (
            len(self) < 1 
            or len(self) > 2
            or not isinstance(self.words[-1], _type)
            or (len(self) == 2 and not isinstance(self.words[0], Prefix))
        )


class Rule(BaseModel):
    subject: Ramble
    verb: Verb
    object: Ramble
    
    def is_simplest_form(self) -> bool:
        return len(self.to_simplest_forms()) == 1
    
    def simplest_forms_iter(self) -> Iterator["Rule"]:
        for subj in self.subject.split(by=Conjunction):
            for obj in self.object.split(by=Conjunction):
                yield Rule(subject=subj, verb=self.verb, object=obj)
    
    def to_simplest_forms(self) -> list["Rule"]:
        return list(self.simplest_forms_iter())
    
    
class Square(BaseModel):
    loc: Location
    blocks: list[Block] = Field(default_factory=list[Block])
    
    @classmethod
    def from_block_loc(cls, loc: Location, block: Block | None = None) -> Self:
        return Square(loc=loc, blocks=[block] if block is not None else [])

    @classmethod
    def from_block_xy(cls, x: int, y: int, block: Block | None = None) -> Self:
        return cls.from_block_loc(block=block, loc=Location(x=x, y=y))

# GridLine = Annotated[list[Square], conlist(Square, min_length=1)]
# SquareGrid = Annotated[list[GridLine], conlist(GridLine, min_length=1)]
GridLine = conlist(Square, min_length=1)
SquareGrid = conlist(GridLine, min_length=1)
# SquareGrid = npt.NDArray[Square]

class Board(BaseModel):
    """board has a m x n grid where (x, y) coordinates are used to locate squares
    col x in [0, 1, ..., m-1], row y in [0, 1, ..., n-1]"""
    grid: SquareGrid
        
    @classmethod
    def from_shape(cls, height: int, width: int) -> Self:
        return Board(grid=[[Square.from_block_xy(x=x, y=y) for x in range(width)] for y in range(height)])
    
    @property
    def height(self) -> int:
        return len(self.grid)
    
    @property
    def width(self) -> int:
        assert self.height > 0
        return len(self.grid[0])
    
    def get(self, x: int, y: int) -> Square | None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return None
        return self.row(y=y)[x]
            
    def add_block(self, x: int, y: int, block: Block) -> Block | None:
        square = self.get(x=x, y=y)
        if square is not None and all(b is not block for b in square.blocks):
            square.blocks.append(block)
            return block
        return None
    
    def pop_block(self, x: int, y: int, block: Block) -> Block | None:
        square = self.get(x=x, y=y)
        if square is not None:
            for i, b in enumerate(square.blocks):
                if b is block:
                    square.blocks.pop(i)
                    return block
        return None
    
    def pop_blocks(self, x: int, y: int, block_type: type[Block] = Block) -> list[Block]:
        # remove all blocks of a (sub)type
        square = self.get(x=x, y=y)
        keep = []
        discard = []
        if square is not None:
            for block in square.blocks:
                if isinstance(block, block_type):
                    discard.append(block)
                else:
                    keep.append(block)
        square.blocks = keep
        return discard
    
    def row(self, y: int) -> GridLine:
        assert 0 <= y < self.height
        return self.grid[y]
    
    def rows(self) -> Iterator[GridLine]:
        for row in self.grid:
            yield row
    
    def col(self, x: int) -> GridLine:
        assert 0 <= x < self.width
        return [row[x] for row in self.rows()]

    def cols(self) -> Iterator[GridLine]:
        for x in range(self.width):
            yield self.row(x=x)
    
    def words_iter(self, seq: GridLine) -> Iterator[list[Block]]:
        # given a row or column (contiguous sequence of squares)
        # iterate over all possible maximal contiguous word sequences
        word_blocks_seq = []
        for square in seq:
            word_blocks = [block for block in square.blocks if isinstance(block, WordBlock)]
            if len(word_blocks) > 0:
                word_blocks_seq.append(word_blocks_seq)
            else:
                if len(word_blocks_seq) > 0:
                    for combo in product(*word_blocks_seq):
                        yield combo
                word_blocks_seq = []
        if len(word_blocks_seq) > 0:
            for combo in product(*word_blocks_seq):
                yield combo
            
    def rules_iter(self, in_simplest_forms: bool = True) -> Iterator[Rule]:
        for seq in chain(self.rows(), self.cols()):
            for words in self.words_iter(seq):
                for rule in Ramble(words=words).rules_iter():
                    if not in_simplest_forms:
                        yield rule
                    else:
                        for simple_rule in rule.to_simplest_forms():
                            yield simple_rule
                    
    def rules(self, in_simplest_forms: bool = True) -> list[Rule]:
        return list(self.rules_iter(in_simplest_forms=in_simplest_forms))
    
    def rules_with_iter(self, subj_type: type[Noun] | None = None, verb_type: type[Verb] | None = None, obj_type: type[Adjective | Noun] | None = None) -> Iterator[Rule]:
        for rule in self.rules_iter(in_simplest_forms=True):
            print(f'{rule=}')
            if subj_type is not None:
                # assert rule.subject.is_simple()
                words = rule.subject.split(by=Conditional)            
                if not any(isinstance(word, obj_type) for word in words):
                    continue

            if verb_type is not None:
                if not isinstance(rule.verb, verb_type):
                    continue
            
            if obj_type is not None:
                # assert rule.object.is_simple()
                word = rule.object.words[-1]
                if not isinstance(word, obj_type):
                    continue
            
            yield rule
    
    def noun_phrases_is_you_iter(self) -> Iterator[Ramble]:
        for rule in self.rules_with_iter(verb_type=Is, obj_type=You):
            yield rule.subject

    def noun_phrases_is_win_iter(self) -> Iterator[Ramble]:
        for rule in self.rules_with_iter(verb_type=Is, obj_type=Win):
            yield rule.subject
    
    def images_is_you_iter(self) -> Iterator[ImageBlock]:
        # TODO: figure out what we return: image block location? square? image itself?
        for noun_phrases in self.noun_phrases_is_you_iter():
            # TODO: this is where logic tied to conditional/subject nouns comes into play
            yield None
        
            
    def __repr__(self) -> str:
        left_sep = '['
        right_sep = ']'
        horiz_sep = '\n'
        intra_col_sep = '|'
        top_sep = '_'
        bottom_sep = '-'
        row_strs = []
        max_width = 0
        for row in reversed(list(self.rows())):
            col_strs = []
            for col in row:
                col_str = intra_col_sep.join(b.__repr__() for b in col.blocks)
                col_strs.append(col_str)
            row_str = f'{right_sep}{left_sep}'.join(col_strs)
            if len(row_str) > max_width:
                max_width = len(row_str)
            row_strs.append(row_str)
        full_str = f'{right_sep}{horiz_sep}{left_sep}'.join(row_strs)
        return horiz_sep + top_sep*(max_width+2) + horiz_sep + left_sep + full_str + right_sep + horiz_sep + bottom_sep*(max_width+2)

class Level(BaseModel):
    # history: Annotated[list[Board], conlist(min_length=1)]
    history: conlist(Board, min_length=1)
    
    @property
    def board(self) -> Board:
        return self.history[-1]
    
    @classmethod
    def from_board(cls, board: Board) -> Self:
        return Level(history=[board])
    
    def rules_iter(self) -> Iterator[Rule]:
        return self.board.rules_iter()
        
    def forward(self, d: Direction) -> None:
        raise NotImplementedError()
    
    def backward(self) -> None:
        if len(self.history) > 1:
            self.history.pop()
            
    def step(self, pressed: Direction | None) -> None:
        if isinstance(pressed, Direction):
            self.forward(pressed)
        else:
            self.backward()
            
    
if __name__ == "__main__":
    s = Ramble(words=[You(), Is(), Baba(), On(), Robot(), Is(), Push(), And(), You(), Is(), Baba()])
    print(f'{s=}')
    r = s.to_rule()
    print(f'{r=}')
    rsf = r.to_simplest_forms()
    print(f'{len(rsf)=}')
    for r in rsf:
        print(f'{r.is_simplest_form()=}')
        print(f'{r=}')
    
    d = Direction.DOWN
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.UP
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.LEFT
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.RIGHT
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')
    d = Direction.STAY
    print(f'{d=}, {d._name_=}, {d.as_loc()=}')

    loc = Location(x=0, y=0)
    print(f'{loc=}')
    new_loc = loc + Direction.DOWN
    print(f'{new_loc=}')
    # assert (new_loc.x == loc.x) and (new_loc.y == loc.y - 1)
    
    print(f'{(Baba() == Baba())=}')
    print(f'{(Baba() is Baba())=}')
    
    board = Board.from_shape(height=5, width=6)
    board.add_block(y=1, x=1, block=Baba())
    board.add_block(y=1, x=2, block=Is())
    board.add_block(y=1, x=3, block=You())
    print(f'{board=}')
    print(f'{list(board.noun_phrases_is_you_iter())=}')
    print(f'{list(board.noun_phrases_is_win_iter())=}')
    level = Level.from_board(board=board)
    print(f'{level=}')
