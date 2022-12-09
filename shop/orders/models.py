from django.db import models

from shop.deliveries.models import Deliveries
from shop.products.models import Products
from shop.susers.models import SUsers


class Orders(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now=True)

    product_id = models.ForeignKey(Products, on_delete=models.CASCADE)
    user_id = models.ForeignKey(SUsers, on_delete=models.CASCADE)
    delivery_id = models.ForeignKey(Deliveries, on_delete=models.CASCADE)

    class Meta:
        db_table = "shop_orders"

    def __str__(self):
        return f'{self.pk} {self.id} {self.created_at}'
