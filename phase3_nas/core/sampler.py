import random

from .search_space import SEARCH_SPACE


class ArchitectureSampler:

    def __init__(self):

        self.search_space = SEARCH_SPACE

    def sample(self):

        architecture = {

            "channels": random.choice(

                self.search_space["channels"]

            ),

            "num_rrdb_blocks": random.choice(

                self.search_space["num_rrdb_blocks"]

            )

        }

        return architecture