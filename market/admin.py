from django.contrib import admin

from .models import MarketPrediction, MarketCandle


@admin.register(MarketPrediction)
class MarketPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "symbol",
        "current_price",
        "direction",
        "confidence",
        "risk_level",
        "created_at",
    )
    list_filter = ("symbol", "direction", "risk_level")
    search_fields = ("symbol", "insight")
    ordering = ("-created_at",)


@admin.register(MarketCandle)
class MarketCandleAdmin(admin.ModelAdmin):
    list_display = (
        "symbol",
        "interval",
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    list_filter = ("symbol", "interval")
    search_fields = ("symbol",)
    ordering = ("-time",)