# -*- coding: utf-8 -*-
#!/usr/env python3

# Copyright (C) 2017 Gaby Launay

# Author: Gaby Launay  <gaby.launay@tutanota.com>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

""" Python tool to make bibliographic networks """

__author__ = "Gaby Launay"
__copyright__ = "Gaby Launay 2017"
__license__ = "GPL3"
__email__ = "gaby.launay@tutanota.com"
__status__ = "Development"

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
        self.author_list = []

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
                              nrows=nmb_to_import, encoding="ISO8859",
                              error_bad_lines=False,
                              warn_bad_lines=True)
        self.db.reset_index()
        # separate authors
        self.db['Authors'] = self.db.apply(self._split_authors, axis=1)
        # Replace missing values
        self.db['Cited by'].fillna(0, inplace=True)
        # Updat author list
        self.update_author_list()

    def clean(self, min_citations=10):
        "Remove some entries"
        len_bef = len(self.db)
        self.db.drop(self.db[self.db["Cited by"] < min_citations].index,
                     inplace=True)
        len_after = len(self.db)
        print("    Removed {} articles, {} remaining".format(len_bef-len_after,
                                                            len_after))
        self.update_author_list()
        self._auth_betw = None

    def update_author_list(self):
        auths = list(set(np.concatenate(self.db['Authors'].values)))
        self.author_list = np.sort(auths)

    @property
    def author_betweeness(self):
        "Compute authors betweness"
        # If already computed, just return it
        if self._auth_betw is not None and \
                self._auth_betw_computed_from == len(self.db):
            return self._auth_betw
        # else compute it
        self._auth_betw_computed_from = len(self.db)
        auth_betw = {auth: {}
                     for auth in self.author_list}
        for auths in self.db['Authors']:
            # skip if only one author
            if len(auths) == 1:
                continue
            # Loop on authors couples
            for i1, auth1 in enumerate(auths):
                for auth2 in auths[i1+1::]:
                    keys = auth_betw.keys()
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
        nmbcits = {}
        for _, art in self.db.iterrows():
            auths = art['Authors']
            nmbcit = int(art['Cited by'])
            for auth in auths:
                if auth in nmbcits.keys():
                    nmbcits[auth] += nmbcit
                else:
                    nmbcits[auth] = nmbcit
        return nmbcits

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

    def write_author_list(self, filepath):
        with open(filepath, "w") as f:
            data = ['{}: {}\n'.format(i, auth)
                    for i, auth in enumerate(self.author_list)]
            f.writelines(data)

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
        auths = self.author_list
        self.graph.add_vertex(len(auths))
        # add links
        auth2ind = {auths[i]: i
                    for i in range(len(auths))}
        abet = []
        authbet = copy.deepcopy(self.author_betweeness)
        for auth in auths:
            for col, weight in authbet[auth].items():
                if col == auth:
                    continue
                self.graph.add_edge(auth2ind[auth], auth2ind[col])
                del authbet[col][auth]  # ensure that edges are not doubled
                abet.append(weight)
        # add properties
        cb = self.graph.new_edge_property("int", abet)
        self.graph.edge_properties['weight'] = cb
        # layout
        if layout == "arf":
            self.layout_pos = arf_layout(self.graph,
                                         weight=self.graph.ep.weight,
                                         pos=self.layout_pos,
                                         max_iter=10000)
        elif layout == "sfpd":
            self.layout_pos = sfdp_layout(self.graph,
                                          eweight=self.graph.ep.weight,
                                          pos=self.layout_pos)
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

    def display_article_graph(self, out="graph.pdf", min_size=1,
                              max_size=10, indice=False):
        """Display an article graph"""
        cb = np.log(np.array(self.graph.vp.nmb_citation.a)+2)
        ms = cb/max(cb)*(max_size - min_size) + min_size
        ms = self.graph.new_vertex_property('float', ms)
        graph_draw(self.graph, pos=self.layout_pos, output=out,
                   vertex_size=ms,
                   vertex_fill_color=self.graph.vp.nmb_citation,
                   vcmap=plt.cm.viridis)

    def display_author_graph(self, out="graph.pdf", min_size=1, max_size=10,
                             indice=False):
        """Display an author graph"""
        auths = self.author_list
        nc = self.get_total_citation()
        nc = [int(nc[auth]) for auth in auths]
        na = self.get_auth_nmb_of_art()
        na = [int(na[auth]) for auth in auths]
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
        # Get vertex display order
        vorder = np.argsort(nc)
        # Get index
        if indice:
            text = range(len(vorder))
            textg = self.graph.new_vertex_property('string', text)
        else:
            textg = None
        # plot
        ncg = self.graph.new_vertex_property('float', nc)
        nag = self.graph.new_vertex_property('int', na)
        vorderg = self.graph.new_vertex_property('int', vorder)
        weightg = self.graph.new_edge_property('float', weight)
        self.graph.vp['nmb_citation'] = ncg
        graph_draw(self.graph, pos=self.layout_pos, output=out,
                   vertex_fill_color=nag, vertex_size=ncg,
                   edge_pen_width=weightg, vertex_text=textg,
                   vorder=vorderg,
                   vertex_text_position=0,
                   vcmap=plt.cm.PuBu)


print("=== Importing")
layouts = ['arf', 'sfpd', 'fr', 'radial']
layouts = ['arf']
for layout in layouts:
    csvi = BiblioNetwork('database_gaby')
    csvi.parse(nmb_to_import=None)
    print("=== Cleaning")
    csvi.clean(min_citations=25)
    # # by author
    # print("=== Make {} graph".format(layout))
    # csvi.make_author_graph(layout=layout)
    # print("=== Display {} graph".format(layout))
    # csvi.display_author_graph(min_size=3, max_size=30,
    #                           out="graph_gaby_{}.pdf".format(layout),
    #                           indice=True)
    # csvi.write_author_list("index_authors_gaby.txt")
    # by article
    print("=== Make {} graph".format(layout))
    csvi.make_article_graph(layout=layout)
    print("=== Display {} graph".format(layout))
    csvi.display_article_graph(min_size=3, max_size=30,
                               out="graph_gaby_{}.pdf".format(layout),
                               indice=True)
