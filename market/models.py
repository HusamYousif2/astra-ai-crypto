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


class MarketCandle(models.Model):
    INTERVAL_CHOICES = [
        ("1h", "1 Hour"),
        ("4h", "4 Hours"),
        ("1d", "1 Day"),
    ]

    symbol = models.CharField(max_length=10)
    interval = models.CharField(max_length=5, choices=INTERVAL_CHOICES, default="1h")
    time = models.DateTimeField()

    open = models.FloatField()
    high = models.FloatField()
    low = models.FloatField()
    close = models.FloatField()

    volume = models.FloatField(default=0.0)
    quote_volume = models.FloatField(default=0.0)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["symbol", "interval", "time"]
        constraints = [
            models.UniqueConstraint(
                fields=["symbol", "interval", "time"],
                name="unique_market_candle_symbol_interval_time",
            )
        ]
        indexes = [
            models.Index(fields=["symbol", "interval", "time"]),
        ]

    def __str__(self):
        return f"{self.symbol} {self.interval} {self.time}"