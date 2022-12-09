from django.db import models

from moviee.cinemas.models import Cinemas


class Theaters(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=10)
    seat = models.IntegerField()

    cinema_id = models.ForeignKey(Cinemas, on_delete=models.CASCADE)

    class Meta:
        db_table = "movie_theaters"

    def __str__(self):
        return f'{self.pk} {self.id} {self.title} {self.seat}'
