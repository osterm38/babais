from babais_alg.schema import *


class TestSentenceBabaIsYou:
    words = [Baba(), Is(), You()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_rule(self):
        sentence = Sentence(words=self.words)
        assert sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_self(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words == rule.words
    

class TestSentenceYouIsBaba:
    words = [You(), Is(), Baba()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_none(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert rule is None


class TestSentenceBabaIsYouBabaIsYou:
    words = [Baba(), Is(), You(), Baba(), Is(), You()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_first_half(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words[:3] == rule.words
    

class TestSentenceBabaIsYouYouIsBaba:
    words = [Baba(), Is(), You(), You(), Is(), Baba()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_first_half(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words[:3] == rule.words
    

class TestSentenceYouIsBabaBabaIsYou:
    words = [You(), Is(), Baba(), Baba(), Is(), You()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_last_half(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words[3:] == rule.words


class TestSentenceIsBabaBabaIsPushPushIs:
    words = [Is(), Baba(), Baba(), Is(), Push(), Push(), Is()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_part(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words[2:5] == rule.words
    

class TestSentenceYouIsBabaAndRobotIsPushAndYouIsBaba:
    words = [You(), Is(), Baba(), And(), Robot(), Is(), Push(), And(), You(), Is(), Baba()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_part(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert sentence.words[2:9] == rule.words
    

class TestSentenceYouIsBabaAndYouIsPushAndYouIsBaba:
    words = [You(), Is(), Baba(), And(), You(), Is(), Push(), And(), You(), Is(), Baba()]
    
    def test_sentence_init(self):
        sentence = Sentence(words=self.words)
        assert sentence.words == self.words

    def test_sentence_is_not_rule(self):
        sentence = Sentence(words=self.words)
        assert not sentence.is_rule()
        
    def test_sentence_to_leftmost_rule_is_none(self):
        sentence = Sentence(words=self.words)
        rule = sentence.to_leftmost_rule()
        assert rule is None
    
class TestLocation:
    x = 0
    y = 0
    
    def test_loc_init(self):
        loc = Location(x=self.x, y=self.y)
        assert self.x == loc.x and self.y == loc.y
    
    def test_loc_down(self):
        loc = Location(x=self.x, y=self.y)
        new_loc = loc + Direction.DOWN
        assert (new_loc.x == loc.x) and (new_loc.y == loc.y - 1)
    
    def test_loc_up(self):
        loc = Location(x=self.x, y=self.y)
        new_loc = loc + Direction.UP
        assert (new_loc.x == loc.x) and (new_loc.y == loc.y + 1)
    
    def test_loc_left(self):
        loc = Location(x=self.x, y=self.y)
        new_loc = loc + Direction.LEFT
        assert (new_loc.x == loc.x - 1) and (new_loc.y == loc.y)
        
    def test_loc_up(self):
        loc = Location(x=self.x, y=self.y)
        new_loc = loc + Direction.RIGHT
        assert (new_loc.x == loc.x + 1) and (new_loc.y == loc.y)
    
    def test_loc_stay(self):
        loc = Location(x=self.x, y=self.y)
        new_loc = loc + Direction.STAY
        assert (new_loc.x == loc.x) and (new_loc.y == loc.y)