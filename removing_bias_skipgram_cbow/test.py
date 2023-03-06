import random
import unittest
import code
import torch


class TestWordEmb(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestWordEmb, self).__init__(*args, **kwargs)
        self.text = 'dogs and cats are playing'.split()
        self.test = [[414, 1907, 523, 2231],
                     [688, 789, 299, 1909],
                     [1417, 343, 1286, 7],
                     [860, 1043, 1080, 859],
                     [596, 517, 1650, 1193],
                     [1767, 1395, 2462, 1888],
                     [238, 1350, 1856, 1153],
                     [842, 909, 66, 1269],
                     [834, 1105, 543, 2116],
                     [2057, 43, 1875, 340],
                     [2267, 1186, 403, 305],
                     [11, 1846, 1144, 326],
                     [487, 655, 121, 844],
                     [1429, 172, 1515, 1458],
                     [1592, 913, 2049, 624],
                     [875, 2046, 1864, 1835],
                     [405, 1516, 859, 1354],
                     [164, 2288, 1698, 1386],
                     [585, 457, 850, 1471],
                     [2040, 957, 1742, 1131],
                     [1107, 2147, 361, 1156],
                     [2283, 1029, 387, 1985],
                     [214, 2039, 1719, 2133],
                     [1952, 345, 986, 1761],
                     [1898, 672, 2385, 314],
                     [126, 1667, 160, 1998],
                     [1900, 658, 724, 289],
                     [2394, 1668, 1716, 1381],
                     [174, 2398, 1842, 866],
                     [1020, 2459, 809, 355],
                     [1420, 372, 94, 933],
                     [1679, 2398, 1464, 54]]
        self.test = torch.Tensor(self.test)

    def testCurPairs(self):
        sur, cur = code.build_current_surrounding_pairs(self.text, 1)
        print(sur, cur)

    def testExpandPairs(self):
        sur, cur = code.expand_surrounding_words(*code.build_current_surrounding_pairs(self.text, 1))
        print(sur, cur)

    def testCBow(self):
        indices_list = [self.text] * 3
        print(code.cbow_preprocessing(indices_list, 1))

    def testSkip(self):
        indices_list = [self.text] * 3
        print(code.skipgram_preprocessing(indices_list, 1))

    def testSimilar(self):
        tens_1 = torch.tensor([[0.2245, 0.2959,
                                0.3597, 0.6766],
                               [-2.2268, 0.6469,
                                0.3765, 0.7898],
                               [0.4577, 0.2959,
                                0.4699, 0.2389]])

        # define second 2D tensor
        word_embed = torch.tensor([[0.2246, 0.2959,
                                    0.3597, 0.6766]])
        print(code.compute_topk_similar(word_embed, tens_1, 2))

    def testRetrieve(self):
        model = code.CBOW(1)
        code.retrieve_similar_words(model, 'hi',
                                    {
                                        'hi': 0
                                    },
                                    {
                                        0: 'hi'
                                    })


class TestBias(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBias, self).__init__(*args, **kwargs)
        self.embeds = code.load_glove_embeddings("data/glove/glove.6B.300d.txt")
        self.professions = code.load_professions("data/professions.tsv")
        self.gender_attrs = code.load_gender_attribute_words(
            "data/gender_attribute_words.json"
        )[:10]
        self.words = ['kindergarten teacher', 'dental hygienist', 'speech-language pathologist', 'dental assistant',
         'childcare worker', 'medical records technician', 'secretary', 'medical assistant', 'hairdresser', 'dietitian',
         'vocational nurse', 'teacher assistant', 'paralegal', 'billing clerk', 'phlebotomist', 'receptionist',
         'housekeeper', 'registered nurse', 'bookkeeper', 'health aide', 'taper', 'steel worker',
         'mobile equipment mechanic', 'bus mechanic', 'service technician', 'heating mechanic', 'electrical installer',
         'operating engineer', 'logging worker', 'floor installer', 'roofer', 'mining machine operator', 'electrician',
         'repairer', 'conductor', 'plumber', 'carpenter', 'security system installer', 'mason', 'firefighter',
         'salesperson', 'director of religious activities', 'crossing guard', 'photographer', 'lifeguard',
         'lodging manager', 'healthcare practitioner', 'sales agent', 'mail clerk', 'electrical assembler',
         'insurance sales agent', 'insurance underwriter', 'medical scientist', 'statistician', 'training specialist',
         'judge', 'bartender', 'dispatcher', 'order clerk', 'mail sorter']

    def testSubspace(self):
        print(code.compute_gender_subspace(self.embeds, self.gender_attrs, 2))

    def testProfession(self):
        print(code.compute_profession_embeddings(self.embeds, self.words))

    def testExtreme(self):
        subspace = code.compute_gender_subspace(self.embeds, self.gender_attrs, 1)
        words = ['bartender', 'dispatcher']
        code.compute_extreme_words(words, self.embeds, subspace)

    def testBias(self):
        subspace = code.compute_gender_subspace(self.embeds, self.gender_attrs, 1)
        words = ['bartender', 'dispatcher']
        code.compute_direct_bias(words, self.embeds, subspace)

    def testDeBias(self):
        code.hard_debias(self.embeds, self.gender_attrs, 1)

if __name__ == '__main__':
    unittest.main()
