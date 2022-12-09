from django.db import models

from blog.busers.models import BUsers


class Posts(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now=True)
    updated_at = models.DateTimeField(auto_now_add=True)

    user_id = models.ForeignKey(BUsers, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_posts"

    def __str__(self):
        return f'{self.pk} {self.id} {self.title} {self.content} {self.created_at} {self.updated_at}'
