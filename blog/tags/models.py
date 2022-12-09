from django.db import models

from blog.posts.models import Posts


class Tags(models.Model):
    use_in_migrations = True
    id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=255)

    post_id = models.ForeignKey(Posts, on_delete=models.CASCADE)

    class Meta:
        db_table = "blog_tags"

    def __str__(self):
        return f'{self.pk} {self.id} {self.title}'
