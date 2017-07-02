#! /bin/python

import copy
import itertools
import pdb

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from graph_tool.all import (Graph, arf_layout, fruchterman_reingold_layout,
                            graph_draw, sfdp_layout, radial_tree_layout,
                            planar_layout)


class BiblioNetwork():
    "Bibliography network displayer"

    def __init__(self, filepath):
        self.filepath = filepath
        self.db = None
        self._auth_betw = None
        self._auth_betw_computed_from = 0
        self.layout_pos = None
        self.graph = None

    @staticmethod
    def _split_authors(row):
        "Split authors of the row"
        auth = row['Authors'].split(", ")
        auth = [", ".join(auth[2*i:2*i+2])
                for i in range(int(len(auth)/2))]
        return auth

    def parse(self, nmb_to_import=None):
        "Parse the database csv file"
        # import database
        self.db = pd.read_csv(self.filepath, ",", index_col=False,
                              nrows=nmb_to_import)
        self.db.reset_index()
        # separate authors
        self.db['Authors'] = self.db.apply(self._split_authors, axis=1)
        # Replace missing values
        self.db['Cited by'].fillna(0, inplace=True)

    def clean(self, min_citations=10):
        "Remove some entries"
        self.db.drop(self.db[self.db["Cited by"] < min_citations].index,
                     inplace=True)

    @property
    def author_betweeness(self):
        "Compute authors betweness"
        # If already computed, just return it
        if self._auth_betw is not None and \
                self._auth_betw_computed_from == len(self.db):
            return self._auth_betw
        # else compute it
        self._auth_betw_computed_from = len(self.db)
        auth_betw = {}
        for auths in self.db['Authors']:
            # skip if only one author
            if len(auths) == 1:
                continue
            # Loop on authors couples
            for i1, auth1 in enumerate(auths):
                for auth2 in auths[i1::]:
                    if auth1 == auth2:
                        continue
                    keys = auth_betw.keys()
                    # create author if necessary
                    if auth1 not in keys:
                        auth_betw[auth1] = {}
                    if auth2 not in keys:
                        auth_betw[auth2] = {}
                    # create couple if necessary, or increment
                    if auth2 not in auth_betw[auth1].keys():
                        auth_betw[auth1][auth2] = 1
                    else:
                        auth_betw[auth1][auth2] += 1
                    if auth1 not in auth_betw[auth2].keys():
                        auth_betw[auth2][auth1] = 1
                    else:
                        auth_betw[auth2][auth1] += 1
        self._auth_betw = auth_betw
        return self._auth_betw

    @author_betweeness.setter
    def author_betweeness(self, val):
        raise Exception("You cannot change that")

    def get_total_citation(self):
        """ Return total number of citations for each author"""
        nmbcit = {}
        for _, art in self.db.iterrows():
            auths = art['Authors']
            tmp_nmbcit = int(art['Cited by'])
            for auth in auths:
                if auth in nmbcit.keys():
                    nmbcit[auth] += tmp_nmbcit
                else:
                    nmbcit[auth] = tmp_nmbcit
        return nmbcit

    def get_auth_nmb_of_art(self):
        """ Return number of article for each author"""
        nmbart = {}
        for _, art in self.db.iterrows():
            auths = art['Authors']
            for auth in auths:
                if auth in nmbart.keys():
                    nmbart[auth] += 1
                else:
                    nmbart[auth] = 1
        return nmbart

    def _get_author_publication(self):
        auth2pub = {}
        for _, art in self.db.iterrows():
            for auth in art['Authors']:
                if auth in auth2pub.keys():
                    auth2pub[auth] += [art.name]
                else:
                    auth2pub[auth] = [art.name]
        return auth2pub

    def make_article_graph(self, layout="arf"):
        """Make an article graph"""
        self.graph = Graph(directed=False)
        # add vertex
        self.graph.add_vertex(len(self.db))
        # add properties
        cb = self.graph.new_vertex_property("int", self.db['Cited by'].values)
        self.graph.vertex_properties['nmb_citation'] = cb
        # Add links
        auths = list(self.author_betweeness.keys())
        auth2ind = {auths[i]: i
                    for i in range(len(auths))}
        auth2pub = self._get_author_publication()
        for _, pubs in auth2pub.items():
            if len(pubs) < 2:
                continue
            combis = itertools.combinations(pubs, 2)
            self.graph.add_edge_list(list(combis))
        # layout
        if layout == "arf":
            self.layout_pos = arf_layout(self.graph)
        elif layout == "sfpd":
            self.layout_pos = sfdp_layout(self.graph)
        elif layout == "fr":
            self.layout_pos = fruchterman_reingold_layout(self.graph)
        elif layout == "radial":
            self.layout_pos = radial_tree_layout(self.graph,
                                                 auth2ind['Logan, B.E.'])
        else:
            raise ValueError()

    def make_author_graph(self, layout="arf"):
        """Make an author graph"""
        self.graph = Graph(directed=False)
        # add vertex
        auths = list(self.author_betweeness.keys())
        self.graph.add_vertex(len(auths))
        # add links
        auth2ind = {auths[i]: i
                    for i in range(len(auths))}
        abet = []
        authbet = copy.deepcopy(self.author_betweeness)
        for auth in auths:
            for col, weight in authbet[auth].items():
                self.graph.add_edge(auth2ind[auth], auth2ind[col])
                del authbet[col][auth]  # ensure that edges are not doubled
                abet.append(weight)
        # add properties
        cb = self.graph.new_edge_property("int", abet)
        self.graph.edge_properties['weight'] = cb
        # layout
        if layout == "arf":
            self.layout_pos = arf_layout(self.graph,
                                         weight=self.graph.ep.weight)
        elif layout == "sfpd":
            self.layout_pos = sfdp_layout(self.graph,
                                          eweight=self.graph.ep.weight)
        elif layout == "fr":
            self.layout_pos = fruchterman_reingold_layout(self.graph,
                                                          weight=self.graph.ep.weight,
                                                          circular=True,
                                                          pos=self.layout_pos)
        elif layout == "radial":
            nc = self.get_total_citation()
            main_auth_ind = np.argmax(list(nc.values()))
            main_auth = list(nc.keys())[main_auth_ind]
            self.layout_pos = radial_tree_layout(self.graph,
                                                 auth2ind[main_auth])
        elif layout == "planar":
            self.layout_pos = planar_layout(self.graph)

        else:
            raise ValueError()

    def display_article_graph(self, out="graph.pdf", size=10):
        """Display an article graph"""
        cb = np.log(np.array(self.graph.vp.nmb_citation.a)+2)
        ms = cb/max(cb)*size
        ms = self.graph.new_vertex_property('float', ms)
        graph_draw(self.graph, pos=self.layout_pos, output=out,
                   vertex_size=ms,
                   vertex_fill_color=self.graph.vp.nmb_citation,
                   vcmap=plt.cm.viridis)

    def display_author_graph(self, out="graph.pdf", min_size=1, max_size=10):
        """Display an author graph"""
        nc = self.get_total_citation()
        nc = [int(nc[auth]) for auth in self.author_betweeness.keys()]
        na = self.get_auth_nmb_of_art()
        na = [int(na[auth]) for auth in self.author_betweeness.keys()]
        # normalize citation number
        nc = np.array(nc, dtype=float)
        nc /= np.max(nc)
        nc *= (max_size - min_size)
        nc += min_size
        # normalize edge width
        weight = np.array(self.graph.ep.weight.a, dtype=float)
        weight /= np.max(weight)
        weight *= (1 - 0.1)
        weight += 0.1
        # plot
        ncg = self.graph.new_vertex_property('float', nc)
        nag = self.graph.new_vertex_property('int', na)
        weightg = self.graph.new_edge_property('float', weight)
        self.graph.vp['nmb_citation'] = ncg
        graph_draw(self.graph, pos=self.layout_pos, output=out,
                   vertex_fill_color=nag, vertex_size=ncg,
                   edge_pen_width=weightg,
                   vcmap=plt.cm.PuBu)


print("=== Importing")
csvi = BiblioNetwork('database.csv')
csvi.parse(nmb_to_import=None)
print("=== Cleaning")
csvi.clean(min_citations=30)
layouts = ['sfpd', 'arf', 'fr', 'radial']
for layout in layouts:
    print("=== Make {} graph".format(layout))
    csvi.make_author_graph(layout=layout)
    print("=== Display graph")
    csvi.display_author_graph(min_size=3, max_size=30,
                              out="graph_{}.pdf".format(layout))
