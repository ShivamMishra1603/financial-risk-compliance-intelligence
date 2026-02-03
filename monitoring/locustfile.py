from locust import HttpUser, task, between
import random

class RiskAnalystUser(HttpUser):
    # Simulate a user thinking for 1-5 seconds between requests
    wait_time = between(1, 5)

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(3)
    def analyze_risk(self):
        # Weighted higher (3x) because this is the core value action
        payload = {
            "text": "The company faces significant liquidity risks due to recent market downturns and increased regulatory scrutiny in the semiconductor sector.",
            "query": "What are the primary liquidity risks?"
        }
        self.client.post("/analyze", json=payload)
