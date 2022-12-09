from django.db import models

class Categories(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=30)

    class Meta:
        db_table = "shop_categories"

    def __str__(self):
        return f'{self.pk} {self.id} {self.name}'
