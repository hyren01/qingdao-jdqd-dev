#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging

import utilities
from services import PETRreader, PETRglobals

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s -%(message)s")


def __read_dictionaries(validation=False):
    """
    读取data/dictionaries目录下的匹配规则。
    """
    logging.info('Verb dictionary:' + PETRglobals.VerbFileName)
    verb_path = utilities._get_data(
        'data/dictionaries',
        PETRglobals.VerbFileName)
    PETRreader.read_verb_dictionary(verb_path)

    logging.info('Actor dictionaries:' + str(PETRglobals.ActorFileList))
    for actdict in PETRglobals.ActorFileList:
        actor_path = utilities._get_data('data/dictionaries', actdict)
        PETRreader.read_actor_dictionary(actor_path)

    logging.info('Agent dictionary:' + PETRglobals.AgentFileName)
    agent_path = utilities._get_data('data/dictionaries', PETRglobals.AgentFileName)
    PETRreader.read_agent_dictionary(agent_path)

    logging.info('Discard dictionary:' + PETRglobals.DiscardFileName)
    discard_path = utilities._get_data('data/dictionaries', PETRglobals.DiscardFileName)
    PETRreader.read_discard_list(discard_path)

    if PETRglobals.IssueFileName != "":
        logging.info('Issues dictionary:' + PETRglobals.IssueFileName)
        issue_path = utilities._get_data('data/dictionaries', PETRglobals.IssueFileName)
        PETRreader.read_issue_list(issue_path)


def init_params():
    """
    初始化规则匹配数据。
    """
    logging.info('Using default config file.')
    PETRreader.parse_Config(utilities._get_data('data/config/', 'PETR_config.ini'))
    __read_dictionaries()
