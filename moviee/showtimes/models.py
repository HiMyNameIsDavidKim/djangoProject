from django.db import models

from moviee.cinemas.models import Cinemas
from moviee.movies.models import Movies
from moviee.theaters.models import Theaters


class Showtimes(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    start_time = models.DateField()
    end_time = models.DateField()

    cinema_id = models.ForeignKey(Cinemas, on_delete=models.CASCADE)
    movie_id = models.ForeignKey(Movies, on_delete=models.CASCADE)
    theater_id = models.ForeignKey(Theaters, on_delete=models.CASCADE)

    class Meta:
        db_table = "movie_showtimes"

    def __str__(self):
        return f'{self.pk} {self.id} {self.start_time} {self.end_time}'
