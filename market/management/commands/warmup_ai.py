from django.core.management.base import BaseCommand
from market.tasks import background_ai_training


class Command(BaseCommand):
    help = "Run AI warm-up manually: fetch data, update models, generate predictions, and fill cache."

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING("Starting AI warm-up..."))
        background_ai_training()
        self.stdout.write(self.style.SUCCESS("AI warm-up finished successfully."))
