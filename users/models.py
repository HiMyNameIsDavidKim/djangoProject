from django.db import models

class Users(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=100)
    password = models.CharField(max_length=255)
    created_at = models.DateField(auto_now=True)

    rank = models.IntegerField(default=1)
    point = models.IntegerField(default=0)

    class Meta:
        db_table = "users"

    def __str__(self):
        return f'{self.pk} {self.username}'
