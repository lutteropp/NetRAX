from django.db import models
from django.urls import reverse

from django.db.models.signals import pre_save

from network.logic.RandomNetworkAndTrees import SimulationParameters, simulate_network_and_sequences

# Create your models here.


class Network(models.Model):
    output_base = models.CharField(
        max_length=200, default=None, null=False, blank=False)
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

    def get_absolute_url(self):
        return reverse("network:network-detail", kwargs={"pk": self.pk})

    def total_msa_length(self):
        return self.n_trees * self.sites_per_tree


def construct_simulation_parameters(instance):
    params = SimulationParameters()
    params.output = instance.output_base
    params.speciation_rate = instance.speciation_rate
    params.hybridization_rate = instance.hybridization_rate
    params.pop_size = instance.population_size
    params.time_limit = instance.time_limit
    params.number_trees = instance.n_trees
    params.number_sites = instance.sites_per_tree
    params.ILS = instance.incomplete_lineage_sorting
    return params


def generate_network(sender, instance, *args, **kwargs):
    print(instance.output_base)
    instance.newick_path = instance.output_base + "_network"
    instance.trees_path = instance.output_base + "_trees"
    instance.newick_dendroscope_path = instance.output_base + "_networkDendroscope"
    instance.msa_path = instance.output_base + ".dat"
    params = construct_simulation_parameters(instance)
    simulate_network_and_sequences(params)

pre_save.connect(generate_network, sender=Network)
