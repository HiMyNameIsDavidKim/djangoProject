from django.db import models

from moviee.musers.models import MUsers
from moviee.showtimes.models import Showtimes
from moviee.theaters.models import Theaters


class Theater_tickets(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    x = models.IntegerField()
    y = models.IntegerField()

    user_id = models.ForeignKey(MUsers, on_delete=models.CASCADE)
    showtime_id = models.ForeignKey(Showtimes, on_delete=models.CASCADE)
    theater_id = models.ForeignKey(Theaters, on_delete=models.CASCADE)

    class Meta:
        db_table = "movie_theater_tickets"

    def __str__(self):
        return f'{self.pk} {self.id} {self.x} {self.y}'
