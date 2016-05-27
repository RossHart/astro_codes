# A _consistent_ method for combining the data from 2 catalogues.

from astropy.table import Table, join
import numpy as np
import math
from astropy.coordinates import SkyCoord
from astropy import units as u


def match_sky(reference_data,match_data,reference_radec=['ra','dec'],match_radec=['ra','dec']):
    
    '''---Find the matches between 2 sets of ra+dec points---
    
    Inputs:
    -------
    reference_data: usually the catlogue we wish to match to (eg. galaxies in GZ).
    
    match_data: usually a subsidiary dataset, eg. detections in AFALFA, WISE, ...
    
    reference_radec, match_radec: names of the columns that contain ra+dec (in degrees).
    
    Outputs:
    --------
    ids: 3 column catalogue of 'match index', 'reference index' and 'separations' (in degrees).   
    '''
    
    reference_ra, reference_dec = [np.array(reference_data[i]) for i in reference_radec]
    match_ra, match_dec = [np.array(match_data[i]) for i in match_radec]
    
    reference_coord = SkyCoord(ra=reference_ra*u.degree, dec=reference_dec*u.degree) 
    match_coord = SkyCoord(ra=match_ra*u.degree, dec=match_dec*u.degree)
    idx, sep, _ = match_coord.match_to_catalog_sky(reference_coord)
    match_idx = np.arange(len(match_data))
    ids = Table(np.array([match_idx,idx,sep.arcsecond]).T
                ,names=('match_index','reference_index','separation'))
    
    print('{} galaxies in the reference catalogue'.format(len(reference_data)))
    print('{} galaxies in the match catalogue'.format(len(match_data)))
    print('---> {} matches in total'.format(len(ids)))
    
    return ids


def match_ids(reference_data,match_data,reference_column='id',match_column='id'):
    
    '''
    ---Find the matches between 2 sets of IDs points---
    
    Inputs:
    -------
    reference_data: usually the catlogue we wish to match to (eg. galaxies in GZ).
    
    match_data: usually a subsidiary dataset, eg. detections in AFALFA, WISE, ...
    
    reference_column, match_column: names of the columns that contain the IDs (eg. DR7 ids).
    
    Outputs:
    --------
    ids: 3 column catalogue of 'match index', 'reference index' and 'id'.   
    '''
    
    reference_indices = np.arange(len(reference_data))
    match_indices = np.arange(len(match_data))
    
    reference_table = Table(np.array([reference_indices,reference_data[reference_column]]).T,
                            names=('reference_index','id'))
    match_table = Table(np.array([match_indices,match_data[match_column]]).T,
                        names=('match_index','id'))
    
    ids = join(reference_table, match_table, keys='id')
    
    print('{} galaxies in the reference catalogue'.format(len(reference_data)))
    print('{} galaxies in the match catalogue'.format(len(match_data)))
    print('---> {} matches in toal'.format(len(ids)))
    
    return ids


def keep_good_matches(matches,max_separation=1):
    
    order = np.argsort(matches['separation'])
    ordered_matches = matches[order]
    _, unique_idx = np.unique(matches['reference_index'],return_index=True)
    good_matches = ordered_matches[unique_idx]
    if max_separation != None:
        good_matches = good_matches[good_matches['separation'] <= max_separation]
        
    print('---> {} unique matches of < {} arcsec'.format(len(good_matches),max_separation))
    
    return good_matches


def check_redshift(reference_data,match_data,matches,z_names=['z','z'],max_separation=0.01):
    
    reference_z = reference_data[matches['reference_index'].astype(int)][z_names[0]]
    match_z = match_data[matches['match_index'].astype(int)][z_names[1]]
    delta_z = np.abs(reference_z-match_z)
    redshift_ok = delta_z <= max_separation
    good_matches = matches[redshift_ok]
    
    print('---> {} unique matches of delta-z < {}'.format(len(good_matches),max_separation))
    
    return good_matches, delta_z


def match_sky_restricted(reference_data,match_data,max_separation=10,max_dz=0.01,
                         reference_xyz=['ra','dec','z'],match_xyz=['ra','dec','z']):
    
    '''
    ---Find the matches between 2 sets of IDs points, with restrictions---
    
    This piece of code only returns the _closest_ match, and only the matches that 
    satidfy a set of matching criteria.
    
    Inputs:
    -------
    reference_data: usually the catlogue we wish to match to (eg. galaxies in GZ).
    
    match_data: usually a subsidiary dataset, eg. detections in AFALFA, WISE, ...
    
    max_separation: maximum separation of objects in arcsec.
    
    max_dz: max difference in redshift. If set to 'None', then no redshift cut 
    is applied.
    
    reference_xyz,match_xyz: columns that contain ra,dec and z of the data. If 
    only 2 strings are passed in either case, no redshift cut is applied.
    
    Outputs:
    --------
    good_ids: 3 column catalogue of 'match index', 'reference index' and 'id'.   
    '''
    
    z_names = [reference_xyz[-1],match_xyz[-1]]
    reference_radec = reference_xyz[:2]
    match_radec = match_xyz[:2]
    
    ids = match_sky(reference_data,match_data,reference_radec,match_radec)
    good_ids = keep_good_matches(ids,max_separation)
    if (max_dz != None) & (len(reference_xyz) == 3) & (len(match_xyz) == 3):
        good_ids, dz = check_redshift(reference_data,match_data,good_ids,z_names,max_dz)
    else:
        print('*No z-cut performed!')
        
    return good_ids


def make_matched_catalogue(reference_data,match_data,ids):
    
    '''
    --- Create a catalogue of 'match' data that aligns perfectly with the reference
    catalogue---
    
    Inputs:
    -------
    reference_data: usually the catlogue we wish to match to (eg. galaxies in GZ).
    
    match_data: usually a subsidiary dataset, eg. detections in AFALFA, WISE, ...
    
    ids: an output from either match_sky(), restricted_match_sky() or match_ids().
    
    Outputs:
    --------
    match_table: table with the _columns_ of match data, matched to the reference 
    data catalogue. The 'mask' column provides simple access to whether the data 
    was matched or not.
    '''
    
    columns = match_data.colnames
    match_table = Table()
    
    mask = np.zeros(len(reference_data),dtype='bool')
    mask[ids['reference_index'].astype(int)] = True
    match_table['mask'] = mask
    
    for c in columns:
        if 'str' not in match_data[c].dtype.name: # only keep data which isn't a string!
            column_data = np.ones(len(reference_data))*(-999)
            column_data[ids['reference_index'].astype(int)] = match_data[c][ids['match_index'].astype(int)]
            match_table[c] = column_data
            
    return match_table