from django.db import models

class SUsers(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    email = models.CharField(max_length=120)
    nickname = models.CharField(max_length=20)
    password = models.CharField(max_length=255)
    point = models.IntegerField()

    class Meta:
        db_table = "shop_users"

    def __str__(self):
        return f'{self.pk} {self.id} {self.email} {self.nickname} {self.password} {self.point}'
