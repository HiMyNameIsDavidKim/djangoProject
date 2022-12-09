from django.db import models

class Cinemas(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=20)
    image_url = models.TextField()
    address = models.CharField(max_length=50)
    detail_address = models.CharField(max_length=30)

    class Meta:
        db_table = "movie_cinemas"

    def __str__(self):
        return f'{self.pk} {self.id} {self.title} {self.image_url} {self.address} {self.detail_address}'
