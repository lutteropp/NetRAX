# Generated by Django 3.1.1 on 2020-09-12 09:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Network',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('n_reticulations', models.TextField()),
                ('n_taxa', models.TextField()),
                ('speciation_rate', models.TextField()),
                ('hybridization_rate', models.TextField()),
                ('time_limit', models.TextField()),
                ('sites_per_tree', models.TextField()),
                ('newick_path', models.TextField()),
                ('msa_path', models.TextField()),
                ('image_path', models.TextField()),
            ],
        ),
    ]