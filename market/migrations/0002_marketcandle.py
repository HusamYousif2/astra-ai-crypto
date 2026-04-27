from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("market", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="MarketCandle",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("symbol", models.CharField(max_length=10)),
                ("interval", models.CharField(choices=[("1h", "1 Hour"), ("4h", "4 Hours"), ("1d", "1 Day")], default="1h", max_length=5)),
                ("time", models.DateTimeField()),
                ("open", models.FloatField()),
                ("high", models.FloatField()),
                ("low", models.FloatField()),
                ("close", models.FloatField()),
                ("volume", models.FloatField(default=0.0)),
                ("quote_volume", models.FloatField(default=0.0)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={
                "ordering": ["symbol", "interval", "time"],
            },
        ),
        migrations.AddIndex(
            model_name="marketcandle",
            index=models.Index(fields=["symbol", "interval", "time"], name="market_mark_symbol__c66d32_idx"),
        ),
        migrations.AddConstraint(
            model_name="marketcandle",
            constraint=models.UniqueConstraint(
                fields=("symbol", "interval", "time"),
                name="unique_market_candle_symbol_interval_time",
            ),
        ),
    ]