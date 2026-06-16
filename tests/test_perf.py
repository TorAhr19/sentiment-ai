from locust import HttpUser, task, between


class SentimentUser(HttpUser):
    """Simule un utilisateur qui appelle l'API SentimentIA."""

    wait_time = between(0.5, 1.5)

    @task(3)
    def predict_positive(self):
        self.client.post("/predict", json={"text": "produit excellent"})

    @task(1)
    def get_stats(self):
        self.client.get("/stats")

    @task(1)
    def predict_negative(self):
        self.client.post("/predict", json={"text": "service horrible"})