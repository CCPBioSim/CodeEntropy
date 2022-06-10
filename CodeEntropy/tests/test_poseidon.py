import os, sys
import MDAnalysis as mda
import datetime as dt
from CodeEntropy.ClassCollection.PoseidonClass import Poseidon
import pandas as pd
import pytest

# def test_poseidon_moleculeLevel():
#     start = dt.datetime.now()
#     wd = os.path.dirname(os.path.abspath(__file__))
#     topo_file = os.path.join(wd,"data/poseidon_example.prmtop")
#     traj_file = os.path.join(wd,"data/poseidon_example.trr")
#     u = mda.Universe(topo_file, traj_file)
#     poseidon_object = Poseidon(container=u, start=2, end=12)
#     result = poseidon_object.run_analysis(level_list = ['moleculeLevel'], verbose=False)
#     data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_moleculeLevel.csv")
#     solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
#     pd.testing.assert_frame_equal(solute_ref, result["moleculeLevel"]["soluteData"], check_dtype=False)
#     data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_moleculeLevel.csv")
#     solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
#     pd.testing.assert_frame_equal(solvent_ref, result["moleculeLevel"]["solventData"], check_dtype=False)
    
# def test_poseidon_residLevel_resname():
#     start = dt.datetime.now()
#     wd = os.path.dirname(os.path.abspath(__file__))
#     topo_file = os.path.join(wd,"data/poseidon_example.prmtop")
#     traj_file = os.path.join(wd,"data/poseidon_example.trr")
#     u = mda.Universe(topo_file, traj_file)
#     poseidon_object = Poseidon(container=u, start=2, end=12)
#     result = poseidon_object.run_analysis(level_list = ['residLevel_resname'], verbose=False)
#     data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_residLevel_resname.csv")
#     solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
#     pd.testing.assert_frame_equal(solute_ref, result["residLevel_resname"]["soluteData"], check_dtype=False)
#     data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_residLevel_resname.csv")
#     solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
#     pd.testing.assert_frame_equal(solvent_ref, result["residLevel_resname"]["solventData"], check_dtype=False)
#     data_dir_contact = os.path.join(wd,"data/resid_contact_matrix_residLevel_resname.csv")
#     contact_ref = pd.read_csv(data_dir_contact, na_values="nan")
#     pd.testing.assert_frame_equal(contact_ref, result["residLevel_resname"]["contactMatrix"], check_dtype=False)

# def test_poseidon_atomLevel():
#     start = dt.datetime.now()
#     wd = os.path.dirname(os.path.abspath(__file__))
#     topo_file = os.path.join(wd,"data/poseidon_example.prmtop")
#     traj_file = os.path.join(wd,"data/poseidon_example.trr")
#     u = mda.Universe(topo_file, traj_file)
#     poseidon_object = Poseidon(container=u, start=2, end=12)
#     result = poseidon_object.run_analysis(level_list = ['atomLevel'], verbose=False)
#     data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_atomLevel.csv")
#     solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
#     pd.testing.assert_frame_equal(solute_ref, result["atomLevel"]["soluteData"], check_dtype=False)
#     data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_atomLevel.csv")
#     solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
#     pd.testing.assert_frame_equal(solvent_ref, result["atomLevel"]["solventData"], check_dtype=False)
#     data_dir_contact = os.path.join(wd,"data/resid_contact_matrix_atomLevel.csv")
#     contact_ref = pd.read_csv(data_dir_contact, na_values="nan")
#     pd.testing.assert_frame_equal(contact_ref, result["atomLevel"]["contactMatrix"], check_dtype=False)

# def test_poseidon_soluteContacts():
#     start = dt.datetime.now()
#     wd = os.path.dirname(os.path.abspath(__file__))
#     topo_file = os.path.join(wd,"data/poseidon_example.prmtop")
#     traj_file = os.path.join(wd,"data/poseidon_example.trr")
#     u = mda.Universe(topo_file, traj_file)
#     poseidon_object = Poseidon(container=u, start=2, end=12)
#     result = poseidon_object.run_analysis(level_list = ['soluteContacts'], verbose=False)
#     data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_soluteContacts.csv")
#     solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
#     pd.testing.assert_frame_equal(solute_ref, result["soluteContacts"]["soluteData"], check_dtype=False)
#     data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_soluteContacts.csv")
#     solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
#     pd.testing.assert_frame_equal(solvent_ref, result["soluteContacts"]["solventData"], check_dtype=False)
@pytest.fixture(scope='session')
def wd():
    wd = os.path.dirname(os.path.abspath(__file__))
    return wd
@pytest.fixture(scope='session')
def poseidon_object(wd):
    topo_file = os.path.join(wd,"data/poseidon_example.prmtop")
    traj_file = os.path.join(wd,"data/poseidon_example.trr")
    u = mda.Universe(topo_file, traj_file)
    poseidon_object = Poseidon(container=u, start=2, end=12)
    return poseidon_object


def test_poseidon_moleculeLevel(poseidon_object, wd):
    result = poseidon_object.run_analysis(level_list = ['moleculeLevel'], verbose=False)
    data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_moleculeLevel.csv")
    solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
    pd.testing.assert_frame_equal(solute_ref, result["moleculeLevel"]["soluteData"], check_dtype=False)
    data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_moleculeLevel.csv")
    solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
    pd.testing.assert_frame_equal(solvent_ref, result["moleculeLevel"]["solventData"], check_dtype=False)
    
def test_poseidon_residLevel_resname(poseidon_object, wd):
    result = poseidon_object.run_analysis(level_list = ['residLevel_resname'], verbose=False)
    data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_residLevel_resname.csv")
    solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
    pd.testing.assert_frame_equal(solute_ref, result["residLevel_resname"]["soluteData"], check_dtype=False)
    data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_residLevel_resname.csv")
    solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
    pd.testing.assert_frame_equal(solvent_ref, result["residLevel_resname"]["solventData"], check_dtype=False)
    data_dir_contact = os.path.join(wd,"data/resid_contact_matrix_residLevel_resname.csv")
    contact_ref = pd.read_csv(data_dir_contact, na_values="nan")
    pd.testing.assert_frame_equal(contact_ref, result["residLevel_resname"]["contactMatrix"], check_dtype=False)

def test_poseidon_atomLevel(poseidon_object, wd):
    result = poseidon_object.run_analysis(level_list = ['atomLevel'], verbose=False)
    data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_atomLevel.csv")
    solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
    pd.testing.assert_frame_equal(solute_ref, result["atomLevel"]["soluteData"], check_dtype=False)
    data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_atomLevel.csv")
    solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
    pd.testing.assert_frame_equal(solvent_ref, result["atomLevel"]["solventData"], check_dtype=False)
    data_dir_contact = os.path.join(wd,"data/resid_contact_matrix_atomLevel.csv")
    contact_ref = pd.read_csv(data_dir_contact, na_values="nan")
    pd.testing.assert_frame_equal(contact_ref, result["atomLevel"]["contactMatrix"], check_dtype=False)

def test_poseidon_soluteContacts(poseidon_object, wd):
    result = poseidon_object.run_analysis(level_list = ['soluteContacts'], verbose=False)
    data_dir_solute = os.path.join(wd,"data/soluteVariables10.0EE_soluteContacts.csv")
    solute_ref = pd.read_csv(data_dir_solute, na_values="nan")
    pd.testing.assert_frame_equal(solute_ref, result["soluteContacts"]["soluteData"], check_dtype=False)
    data_dir_solvent = os.path.join(wd,"data/solventVariables10.0EE_soluteContacts.csv")
    solvent_ref = pd.read_csv(data_dir_solvent, na_values="nan")
    pd.testing.assert_frame_equal(solvent_ref, result["soluteContacts"]["solventData"], check_dtype=False)