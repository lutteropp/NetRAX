# Generated by Django 3.1.1 on 2020-09-12 15:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('network', '0003_auto_20200912_1527'),
    ]

    operations = [
        migrations.AlterField(
            model_name='network',
            name='hybridization_rate',
            field=models.FloatField(default=10.0),
        ),
        migrations.AlterField(
            model_name='network',
            name='n_reticulations',
            field=models.PositiveSmallIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='network',
            name='n_taxa',
            field=models.PositiveSmallIntegerField(default=0),
        ),
        migrations.AlterField(
            model_name='network',
            name='speciation_rate',
            field=models.FloatField(default=20.0),
        ),
        migrations.AlterField(
            model_name='network',
            name='time_limit',
            field=models.FloatField(default=0.1),
        ),
    ]