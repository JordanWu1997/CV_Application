#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# vim: set fileencoding=utf-8

import os
import uniface
from uniface.model_store import verify_model_weights

if __name__ == '__main__':
    model_dir = './models'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    for const in dir(uniface.constants):
        if not 'Weight' in const:
            continue
        model = eval(f'uniface.constants.{const}')
        for element in dir(model):
            if element.startswith('__'):
                break
            model_weight = eval(f'uniface.constants.{const}.{element}')
            try:
                model_path = verify_model_weights(model_weight, root=model_dir)
            except Exception as e:
                print(e)
