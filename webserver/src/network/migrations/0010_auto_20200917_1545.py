# Generated by Django 3.1.1 on 2020-09-17 15:45

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0009_network_trees_path'),
    ]

    operations = [
        migrations.AlterField(
            model_name='network',
            name='output_base',
            field=models.CharField(default=None, max_length=200, unique=True),
        ),
    ]
