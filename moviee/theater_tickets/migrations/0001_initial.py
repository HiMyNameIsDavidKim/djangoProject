# Generated by Django 4.1 on 2022-11-30 07:07

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('showtimes', '0001_initial'),
        ('theaters', '0001_initial'),
        ('musers', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Theater_tickets',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('showtime_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='showtimes.showtimes')),
                ('theater_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='theaters.theaters')),
                ('user_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='musers.musers')),
            ],
            options={
                'db_table': 'movie_theater_tickets',
            },
        ),
    ]
