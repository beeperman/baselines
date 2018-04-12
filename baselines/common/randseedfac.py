class Rand():
    def __init__(self):
        self._seed = 5555

    def seed(self, seedgiven):
        self._seed = seedgiven

    def get_seed(self):
        self._seed += 1
        return self._seed

seeder = Rand()