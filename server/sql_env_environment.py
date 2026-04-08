from models import SqlAction, SqlObservation
import random


class SqlEnvironment:

    def __init__(self):
        self.query = ""
        self.task = "easy"
        self.done = False

    async def reset(self):
        # Select random task
        self.task = random.choice(["easy", "medium", "hard"])

        if self.task == "easy":
            self.query = "SELECT * FROM users"
        elif self.task == "medium":
            self.query = "SELECT name FROM users"
        else:
            self.query = "SELECT * FROM orders JOIN users ON users.id = orders.user_id"

        self.done = False

        return SqlObservation(
            query=self.query,
            task=self.task
        )

    async def step(self, action: SqlAction):
        query = action.optimized_query

        reward = 0.0

        if "SELECT *" not in query:
            reward += 0.4

        if "WHERE" in query:
            reward += 0.3

        if "JOIN" in query:
            reward += 0.3

        reward = min(reward, 1.0)
        done = reward >= 0.7

        return {
        "observation": SqlObservation(query=query, task=self.task),  # ✅ object
        "reward": reward,
        "done": done,
        "info": {}
    }

    def state(self):
        return {
            "query": self.query,
            "task": self.task
        }

    async def step_async(self, action: SqlAction):
        return await self.step(action)

    def close(self):
        pass