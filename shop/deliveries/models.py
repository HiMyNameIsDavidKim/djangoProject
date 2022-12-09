from django.db import models

from shop.susers.models import SUsers


class Deliveries(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=20)
    address = models.CharField(max_length=255)
    detail_address = models.CharField(max_length=255)
    phone = models.CharField(max_length=20)

    user_id = models.ForeignKey(SUsers, on_delete=models.CASCADE)


    class Meta:
        db_table = "shop_deliveries"

    def __str__(self):
        return f'{self.pk} {self.id} {self.username} {self.address} {self.detail_address} {self.phone}'
