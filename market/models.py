from django.db import models


class MarketPrediction(models.Model):
    symbol = models.CharField(max_length=10)
    current_price = models.FloatField()
    direction = models.CharField(max_length=20)
    confidence = models.FloatField()
    insight = models.TextField()
    risk_level = models.CharField(max_length=20)
    nlp_score = models.FloatField(default=0.0)
    max_drawdown = models.FloatField(default=0.0)
    value_at_risk = models.FloatField(default=0.0)
    risk_reward = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.symbol} - {self.direction} - {self.created_at}"