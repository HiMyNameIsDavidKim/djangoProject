from django.db import models

class Movies(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=120)
    director = models.CharField(max_length=20)
    description = models.TextField()
    poster_url = models.TextField()
    running_time = models.IntegerField(default=0)
    age_rating = models.IntegerField(default=0)

    class Meta:
        db_table = "movie_moives"

    def __str__(self):
        return f'{self.pk} {self.id} {self.title} {self.director} {self.description} {self.poster_url} {self.running_time} {self.age_rating}'
