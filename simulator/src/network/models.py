from django.db import models
from django.urls import reverse

from django.db.models.signals import pre_save, pre_delete

from network.logic.RandomNetworkAndTrees import SimulationParameters, simulate_network_and_sequences
import os
# Create your models here.


class Network(models.Model):
    output_base = models.CharField(
        max_length=200, default=None, null=False, blank=False, unique=True)
    n_reticulations = models.PositiveSmallIntegerField(
        default=0, null=True, blank=True)
    n_taxa = models.PositiveSmallIntegerField(default=0, null=True, blank=True)
    speciation_rate = models.FloatField(default=20.0)
    hybridization_rate = models.FloatField(default=10.0)
    population_size = models.FloatField(default=0.01)
    time_limit = models.FloatField(default=0.1)
    n_trees = models.PositiveSmallIntegerField(default=1)
    sites_per_tree = models.PositiveSmallIntegerField(default=100)
    newick_path = models.TextField(default=None, null=True, blank=True)
    trees_path = models.TextField(default=None, null=True, blank=True)
    newick_dendroscope_path = models.TextField(
        default=None, null=True, blank=True)
    msa_path = models.TextField(default=None, null=True, blank=True)
    image_path = models.TextField(default=None, null=True, blank=True)
    incomplete_lineage_sorting = models.BooleanField(
        default=False, null=False, blank=True)
    benchmark_mode = models.BooleanField(
        default=False, null=False, blank=True)

    def get_absolute_url(self):
        return reverse("network:network-detail", kwargs={"pk": self.pk})

    def total_msa_length(self):
        return self.n_trees * self.sites_per_tree


def construct_simulation_parameters(instance):
    params = SimulationParameters()
    params.output = 'data/'+instance.output_base
    params.speciation_rate = instance.speciation_rate
    params.hybridization_rate = instance.hybridization_rate
    params.pop_size = instance.population_size
    params.time_limit = instance.time_limit
    params.number_trees = instance.n_trees
    params.number_sites = instance.sites_per_tree
    params.ILS = instance.incomplete_lineage_sorting
    params.benchmark_mode = instance.benchmark_mode
    return params


def generate_network(sender, instance, *args, **kwargs):
    instance.newick_path = 'data/'+instance.output_base + "_network"
    instance.trees_path = 'data/'+instance.output_base + "_trees"
    instance.newick_dendroscope_path = 'data/'+instance.output_base + "_networkDendroscope"
    instance.msa_path = 'data/'+instance.output_base + ".dat"
	#instance.image_path = 'data/'+instance.output_base * "_graph.png"
    params = construct_simulation_parameters(instance)
    instance.n_taxa, instance.n_reticulations = simulate_network_and_sequences(params)


def delete_network_data(sender, instance, *args, **kwargs):
    if instance.newick_path and os.path.exists(instance.newick_path):
        os.remove(instance.newick_path)
    if instance.trees_path and os.path.exists(instance.trees_path):
        os.remove(instance.trees_path)
    if instance.newick_dendroscope_path and os.path.exists(instance.newick_dendroscope_path):
        os.remove(instance.newick_dendroscope_path)
    if instance.msa_path and os.path.exists(instance.msa_path):
        os.remove(instance.msa_path)
    if instance.image_path and os.path.exists(instance.image_path):
        os.remove(instance.image_path)


pre_save.connect(generate_network, sender=Network)
pre_delete.connect(delete_network_data, sender=Network)
