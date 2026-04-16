from django.apps import AppConfig


class MarketConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "market"

    def ready(self):
        """
        Start the scheduler only when appropriate.
        Keep startup lightweight for production web services like Render.
        """
        from .tasks import start_scheduler

        start_scheduler()