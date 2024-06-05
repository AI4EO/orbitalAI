import os
import random
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import h5py
import numpy as np
import rasterio
import shapely.geometry
import shapely.ops
import utm
from cv2 import BORDER_CONSTANT, INTER_LINEAR, INTER_NEAREST, warpAffine
from scipy.ndimage import convolve
from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.utils.parsing import FeaturesSpecification
from eolearn.io import ExportToTiffTask, ImportFromTiffTask
from eolearn.mask.utils import resize_images
from prisma_constants import (
    DIRECTIONS,
    HSI_BANDS,
    HSI_CW,
    HSI_IRR,
    PRISMA_PIX_SIZE,
    RAND_MEAN,
    RAND_STD,
    RELATIVE_SHIFT,
    ProcessingLevels,
)
from pyproj import CRS
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from sentinelhub import BBox, pixel_to_utm
from tqdm.auto import tqdm

#########################
#    DATA MANAGEMENT    #
#########################


def loadHDF5(inFname: str) -> Tuple[Dict, Dict]:
    """Load data/metadata from standard PRS .he5 file. Only those data/metadata relevant for
    this pre-procestool are loaded here."""

    hf = h5py.File(inFname, "r")

    # READ GLOBAL ATTRIBUTES
    cw_vnir = hf.attrs["List_Cw_Vnir"]
    cw_swir = hf.attrs["List_Cw_Swir"]
    cw_vnir_flags = hf.attrs["List_Cw_Vnir_Flags"]
    cw_swir_flags = hf.attrs["List_Cw_Swir_Flags"]
    fwhm_vnir = hf.attrs["List_Fwhm_Vnir"]
    fwhm_swir = hf.attrs["List_Fwhm_Swir"]
    offset_vnir = hf.attrs["Offset_Vnir"]
    offset_swir = hf.attrs["Offset_Swir"]
    scalefac_vnir = hf.attrs["ScaleFactor_Vnir"]
    scalefac_swir = hf.attrs["ScaleFactor_Swir"]
    sun_azimuth = hf.attrs["Sun_azimuth_angle"]
    sun_zenith = hf.attrs["Sun_zenith_angle"]

    # DATA FIELDS - HCO
    vnir_cube = hf.get("HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/VNIR_Cube")
    swir_cube = hf.get("HDFEOS/SWATHS/PRS_L1_HCO/Data Fields/SWIR_Cube")
    vnir_cube = np.array(vnir_cube)
    swir_cube = np.array(swir_cube)
    vnir_cube = np.rot90(
        np.moveaxis(vnir_cube, [0, 1, 2], [0, 2, 1]), -1, (0, 1)
    )  # re-arrange cube axis as [ALT,ACT,BANDS]
    swir_cube = np.rot90(
        np.moveaxis(swir_cube, [0, 1, 2], [0, 2, 1]), -1, (0, 1)
    )  # re-arrange cube axis as [ALT,ACT,BANDS]

    # GELOCATION FIELDS - HCO
    lat_grid = hf.get("HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Latitude_VNIR")
    lon_grid = hf.get("HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields/Longitude_SWIR")
    lat_grid = np.array(lat_grid)
    lon_grid = np.array(lon_grid)
    lat_grid = np.rot90(lat_grid, -1, (0, 1))  # re-arrange grid axis as [ALT,ACT]
    lon_grid = np.rot90(lon_grid, -1, (0, 1))  # re-arrange grid axis as [ALT,ACT]

    # Delivery outputs
    data = {}
    attr = {}

    attr["cw_vnir"] = cw_vnir
    attr["cw_swir"] = cw_swir
    attr["cw_vnir_flags"] = cw_vnir_flags
    attr["cw_swir_flags"] = cw_swir_flags
    attr["fwhm_vnir"] = fwhm_vnir
    attr["fwhm_swir"] = fwhm_swir
    attr["offset_vnir"] = offset_vnir
    attr["offset_swir"] = offset_swir
    attr["scalefac_vnir"] = scalefac_vnir
    attr["scalefac_swir"] = scalefac_swir
    attr["sun_azimuth"] = sun_azimuth
    attr["sun_zenith"] = sun_zenith

    data["vnir_cube"] = vnir_cube
    data["swir_cube"] = swir_cube
    data["lat_grid"] = lat_grid
    data["lon_grid"] = lon_grid

    return data, attr


def PRSdata_DN2Rad(data: Dict, attr: Dict) -> Dict:
    """Converts VNIR and SWIR cubes units from Digital Numbers (DN) stored as UINT16,
    to physical Radiance Units (W/m2/sr/um) stored as FLOAT32"""

    # Ingest useful data/metadata
    vnir_cube_DN = data["vnir_cube"]
    swir_cube_DN = data["swir_cube"]

    offset_vnir = attr["offset_vnir"]
    offset_swir = attr["offset_swir"]
    scalefac_vnir = attr["scalefac_vnir"]
    scalefac_swir = attr["scalefac_swir"]

    # Rescale VNIR, SWIR cubes according to scale factors
    vnir_cube_rad = vnir_cube_DN.astype("float32") / scalefac_vnir + offset_vnir
    swir_cube_rad = swir_cube_DN.astype("float32") / scalefac_swir + offset_swir

    # Delivery outputs
    data["vnir_cube_rad"] = vnir_cube_rad
    data["swir_cube_rad"] = swir_cube_rad

    return data


def gdaltranslateWrapper(inFname: str, outFname: str, latgrid: np.ndarray, longrid: np.ndarray):
    """Run `gdal_translate` to add geolocation information to tiffs in WGS84 CRS"""
    # SETUP
    nrows = latgrid.shape[0]
    ncols = latgrid.shape[1]
    EPSGcode = 4326  # WGS84

    TL = (0, 0)
    TR = (0, ncols - 1)
    BR = (nrows - 1, ncols - 1)
    BL = (nrows - 1, 0)

    (TLlon, TLlat) = (longrid[TL], latgrid[TL])
    (TRlon, TRlat) = (longrid[TR], latgrid[TR])
    (BRlon, BRlat) = (longrid[BR], latgrid[BR])
    (BLlon, BLlat) = (longrid[BL], latgrid[BL])

    # build command
    command = (
        f"gdal_translate -of GTiff -a_srs EPSG:{EPSGcode} -gcp {TL[1]} {TL[0]} {TLlon} {TLlat}"
        f" -gcp {TR[1]} {TR[0]} {TRlon} {TRlat}"
        f" -gcp {BR[1]} {BR[1]} {BRlon} {BRlat}"
        f" -gcp { BL[1]} {BL[0]} {BLlon} {BLlat} {inFname} {outFname}"
    )

    # execute
    print(f"Executing command: \n{command}")
    result = subprocess.run(command.split(" "), check=True, stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))


def gdalwarpWrapper(inFname: str, outFname: str, EPSGcode: str, xres: int = 30, yres: int = 30):
    """Run gdalwrap to project the raster image to a UTM CRS"""
    # build command
    command = f"gdalwarp -t_srs EPSG:{EPSGcode} -tr {xres:.3f} {yres:.3f} -overwrite {inFname} {outFname}"

    # execute
    print(f"Executing command: \n{command}")
    result = subprocess.run(command.split(" "), check=True, stdout=subprocess.PIPE)
    print(result.stdout.decode("utf-8"))


#########################
#        EO Tasks       #
#########################


class ReadAndPreprocessPrismaTask(EOTask):
    """Task to import the PRISMA data into Eopatches for further processing"""

    def get_timestamp_from_filename(self, filename: str) -> datetime:
        """Retrieve acquisition date from PRISMA filename"""
        ts = os.path.basename(filename).split("_")[4]

        return datetime(
            year=int(ts[0:4]),
            month=int(ts[4:6]),
            day=int(ts[6:8]),
            hour=int(ts[8:10]),
            minute=int(ts[10:12]),
            second=int(ts[12:14]),
        )

    def execute(self, eopatch: EOPatch = None, *, filename: str) -> EOPatch:
        """The following steps are executed
        * read HDF5 file with PRISMA's data and metadata
        * convert Digital Numbers to Radiance
        * save cube to tiff files
        * use gdal_translate to add geolocation information
        * use gdalwarp to project from WGS84 to UTM CRS
        * import tiffs into an EOPatch

        :param eopatch: Existing EOPatch where PRISMA data will be added. If None, a new EOPatch is created.
        :param filename: Path to input PRISMA tile.
        """
        # IMPORT PRISMA LEVEL-1 DATA and metadata
        dataPRS, attrPRS = loadHDF5(filename)

        # HYP-CUBES from DN to RADIANCE
        dataPRS = PRSdata_DN2Rad(dataPRS, attrPRS)
        vnirRAD = dataPRS["vnir_cube_rad"]
        swirRAD = dataPRS["swir_cube_rad"]
        latgrid = dataPRS["lat_grid"]
        longrid = dataPRS["lon_grid"]

        # make output folders
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save intermediate TIFF files for hyperspectral cubes
            # VNIR
            outTiffFname_vnir = os.path.join(tmpdirname, "vnirCube.tiff")
            outTiffFname_swir = os.path.join(tmpdirname, "swirCube.tiff")

            with rasterio.open(
                outTiffFname_vnir,
                "w",
                driver="GTiff",
                height=vnirRAD.shape[0],
                width=vnirRAD.shape[1],
                count=vnirRAD.shape[2],
                dtype=vnirRAD.dtype,
            ) as dst:
                for i in range(vnirRAD.shape[2]):
                    dst.write(vnirRAD[..., i], i + 1)

            with rasterio.open(
                outTiffFname_swir,
                "w",
                driver="GTiff",
                height=swirRAD.shape[0],
                width=swirRAD.shape[1],
                count=swirRAD.shape[2],
                dtype=swirRAD.dtype,
            ) as dst:
                for i in range(swirRAD.shape[2]):
                    dst.write(swirRAD[..., i], i + 1)

            # Save hyperspectral cube as geotiff adding corner info (using gdal_translate)
            outGeoTiffFname_vnir = os.path.join(tmpdirname, "vnirCubeWithCorners.tiff")
            gdaltranslateWrapper(outTiffFname_vnir, outGeoTiffFname_vnir, latgrid, longrid)

            outGeoTiffFname_swir = os.path.join(tmpdirname, "swirCubeWithCorners.tiff")
            gdaltranslateWrapper(outTiffFname_swir, outGeoTiffFname_swir, latgrid, longrid)

            # Compute UTM Zone for pixel @center of Lat/Lon grids
            centerLat = latgrid[latgrid.shape[0] // 2, latgrid.shape[1] // 2]
            centerLon = longrid[longrid.shape[0] // 2, longrid.shape[1] // 2]
            (centerEast, centerNorth, UTMZoneNumber, UTMZoneLetter) = utm.from_latlon(centerLat, centerLon)

            # use PROJ dictionary, assuming a default WGS84
            isSouth = True if centerLon < 0 else False
            crs = CRS.from_dict({"proj": "utm", "zone": UTMZoneNumber, "south": isSouth})
            EPSGcode = crs.to_authority()[1]

            # Warp hyperspectral cube in UTM projection, w/ custom pixel spacing (IF NEEDED)

            outTiffWarpedFname_vnir = os.path.join(tmpdirname, f"vnirCubeWarped_EPSG_{EPSGcode}.tiff")
            gdalwarpWrapper(
                outGeoTiffFname_vnir,
                outTiffWarpedFname_vnir,
                EPSGcode,
                PRISMA_PIX_SIZE,
                PRISMA_PIX_SIZE,
            )

            outTiffWarpedFname_swir = os.path.join(tmpdirname, f"swirCubeWarped_EPSG_{EPSGcode}.tiff")
            gdalwarpWrapper(
                outGeoTiffFname_swir,
                outTiffWarpedFname_swir,
                EPSGcode,
                PRISMA_PIX_SIZE,
                PRISMA_PIX_SIZE,
            )

            # covert tiff to EOPatch
            TiffToEopatchTask = ImportFromTiffTask(feature=(FeatureType.DATA, "vnir"), folder=tmpdirname, use_vsi=False)
            eopatch = TiffToEopatchTask.execute(filename=f"vnirCubeWarped_EPSG_{EPSGcode}.tiff")

            TiffToEopatchTask = ImportFromTiffTask(feature=(FeatureType.DATA, "swir"), folder=tmpdirname, use_vsi=False)
            TiffToEopatchTask.execute(eopatch, filename=f"swirCubeWarped_EPSG_{EPSGcode}.tiff")

            eopatch.meta_info = attrPRS
            eopatch.timestamp = [self.get_timestamp_from_filename(filename=filename)]

        return eopatch


class MergeAndSortTask(EOTask):
    def __init__(
        self,
        output_feature: Tuple[FeatureType, str],
        cw_feature: Tuple[FeatureType, str],
    ):
        self.output_feature = self.parse_feature(output_feature, allowed_feature_types=FeatureType.DATA)
        self.cw_feature = self.parse_feature(cw_feature, allowed_feature_types=FeatureType.SCALAR_TIMELESS)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        swir_valid = np.array(eopatch.meta_info["cw_swir_flags"], dtype=bool)
        vnir_valid = np.array(eopatch.meta_info["cw_vnir_flags"], dtype=bool)
        valid = np.concatenate([swir_valid, vnir_valid])

        merged = np.concatenate([eopatch.data["swir"], eopatch.data["vnir"]], axis=-1)

        swir_cw = np.array(eopatch.meta_info["cw_swir"])
        vnir_cw = np.array(eopatch.meta_info["cw_vnir"])
        cw = np.concatenate([swir_cw, vnir_cw])

        cw = cw[valid]
        merged = merged[..., valid]

        sorted_indices = np.argsort(cw)
        eopatch[self.output_feature] = merged[..., sorted_indices]
        eopatch[self.cw_feature] = cw[sorted_indices]
        data_mask = np.sum(eopatch[self.output_feature], axis=-1)[..., np.newaxis] > 0.0
        eopatch.mask["dataMask"] = data_mask.astype(np.uint8)
        return eopatch


class RemoveBandsTask(EOTask):
    def __init__(
        self,
        feature_bands: FeaturesSpecification,
        feature_cw_prisma: FeaturesSpecification,
        min_spectra: float = 400,
        max_spectra: float = 1100,
    ):
        """Remove bands that have central wavelength outside of provided range.

        :param feature_bands: Feature in EOPatch holding the band values.
        :param feature_cw_prisma: Feature in EOPatch with central wavelength.
        :param min_spectra: Min value of range of wavelength.
        :param max_spectra: Max value of range of wavelength.
        """
        self.feature = self.parse_renamed_feature(feature_bands)
        self.feature_cw = self.parse_renamed_feature(feature_cw_prisma)
        self.min_spectra = min_spectra
        self.max_spectra = max_spectra

    def execute(self, eopatch: EOPatch) -> EOPatch:
        cw_prisma = eopatch[self.feature_cw[0]][self.feature_cw[1]]
        cw_mask = (self.min_spectra < cw_prisma) & (cw_prisma < self.max_spectra)

        eopatch[self.feature_cw[0]][self.feature_cw[2]] = cw_prisma[cw_mask]
        eopatch[self.feature[0]][self.feature[2]] = eopatch[self.feature[0]][self.feature[1]][..., cw_mask]

        return eopatch


class SpectralResamplingTask(EOTask):
    def __init__(
        self,
        feature_bands: FeaturesSpecification,
        feature_cw_prisma: FeaturesSpecification,
        sigma: float = 0.72,
    ):
        """Apply spline interpolation to PRISMA bands to HSI central wavelength.

        :param feature_bands: Bands in EOPatch to be interpolated.
        :param feature_cw_prisma: Feature in EOPatch holding the central wavelengths used for interpolation.
        :param sigma: Sigma used to smooth the bands prior to interpolation
        """
        self.feature = self.parse_renamed_feature(feature_bands)
        self.feature_cw = self.parse_feature(feature_cw_prisma)
        self.sigma = sigma

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bands = eopatch[self.feature[0]][self.feature[1]].squeeze(axis=0).copy()

        bands = gaussian_filter1d(bands, self.sigma)
        height, width, nchannels = bands.shape

        bands = bands.reshape(height * width, nchannels)
        bands_new = np.zeros((height * width, len(HSI_CW)))
        cw_prisma = eopatch[self.feature_cw]

        for idx in tqdm(np.arange(height * width)):
            spline = CubicSpline(cw_prisma, bands[idx])
            bands_new[idx, :] = spline(HSI_CW)

        eopatch[self.feature[0]][self.feature[2]] = bands_new.reshape(1, height, width, len(HSI_CW))

        return eopatch


class ResizeTask(EOTask):
    def __init__(
        self,
        features: FeaturesSpecification,
        pixel_size_old: float,
        pixel_size_new: float,
    ):
        self.features = self.parse_renamed_features(features)
        self.pixel_size_old = pixel_size_old
        self.pixel_size_new = pixel_size_new

    def execute(self, eopatch):
        eopatch_new = EOPatch(timestamp=eopatch.timestamp, meta_info=eopatch.meta_info)

        height, width = eopatch.get_spatial_dimension(self.features[0][0], self.features[0][1])

        height_new = height * self.pixel_size_old / self.pixel_size_new
        width_new = width * self.pixel_size_old / self.pixel_size_new

        new_size = (int(height_new), int(width_new))

        for feat_type, feat_name, feat_name_new in self.features:
            eopatch_new[feat_type][feat_name_new] = resize_images(
                eopatch[feat_type][feat_name],
                new_size=new_size,
                interpolation="cubic",
            )

        eopatch_new.bbox = eopatch.bbox

        return eopatch_new


class HSICalculationTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        executable: str,
        calculation: str,
    ):
        """Wapper task to simulate SNR and PSF noise using an external executable binary.

        :param input_feature: Input feature holding radiance values.
        :param output_feature: Output feature with SNR/PSF simulated values.
        :param executable: Path to executable binary.
        :param calculation: Which simulation to execute, i.e. "SNR" or "PSF".
        """
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.executable = executable
        self.calculation = calculation

    def execute(self, eopatch: EOPatch) -> EOPatch:
        with tempfile.TemporaryDirectory(prefix="hsi_calc", suffix=self.calculation) as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")

            # temporary write input feature to numpy
            np.save(input_npy, eopatch[self.input_feature])

            # run exec
            subprocess.run(
                f"{self.executable} {self.calculation} {input_npy} {output_npy}".split(" "),
                check=True,
            )

            # read temp output
            eopatch[self.output_feature] = np.load(output_npy)

        # return eopatch with new feature
        return eopatch


class AlternativeHSICalculationTask(EOTask):
    HSI_BANDS = [f"BAND_{cw}" for cw in np.arange(450, 950, 10)]

    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        snr_feature: Tuple[FeatureType, str],
        snr_values: Dict[str, int],
        l_ref: float,
        psf_feature: Tuple[FeatureType, str],
        psf_kernel: np.array,
    ):
        """Wapper task to simulate dummy SNR and PSF noise for Prisma

        :param input_feature: Input feature holding radiance values.
        :param snr_feature: Output feature with SNR simulated (dummy) values.
        :param psf_feature: Output feature with PSF simulated (dummy) values.
        :param snr_values: A dictionary of SNR values {band: value, ...}.
        :param psf_kernel: PSF 7x7 kernel for HSI
        :param l_ref: Spectral Radiance at Aperture (W/m^2/sr/um), a reference radiance used to generate the specific SNR
        """
        self.input_feature = input_feature
        self.snr_feature = snr_feature
        self.psf_feature = psf_feature

        if all([band in AlternativeHSICalculationTask.SNR_BANDS for band in snr_values.keys()]):
            self.snr_values = snr_values
        else:
            raise Exception("`snr_values` dictionary is missing SNR values for some bands!")

        if psf_kernel.shape == (7, 7):
            self.psf_kernel = psf_kernel
        else:
            raise Exception("`psf_kernel` should have shape (7,7)!")

        self.l_ref = l_ref

    def add_psf(self, eopatch):
        convolved_data = np.concatenate(
            [
                convolve(data_[..., ch], self.psf_kernel, mode="mirror")[..., np.newaxis]
                for data_ in data
                for ch in np.arange(eopatch[self.snr_feature].shape[-1])
            ],
            axis=-1,
        )[np.newaxis, ...]
        eopatch[self.psf_feature] = convolved_data

    def add_snr(self, eopatch):
        radiances = eopatch[self.input_feature]
        random_noise = np.random.normal(size=radiances.shape)

        snr = np.array([self.snr_values[band] for band in AlternativeHSICalculationTask.HSI_BANDS])
        snr = np.reshape(snr, (1, 1, 1, len(snr)))  # t, h, w, d

        noisy_radiances = radiances * (1 + np.sqrt(radiances) * random_noise / (snr * np.sqrt(self.l_ref)))
        eopatch[self.snr_feature] = noisy_radiances

    def execute(self, eopatch: EOPatch) -> EOPatch:
        self.add_snr(eopatch)
        self.add_psf(eopatch)

        return eopatch


def get_shifts(direction: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Compute random shifts."""

    mis_amplitude = np.random.normal(RAND_MEAN, RAND_STD, size=(len(HSI_BANDS),))
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(HSI_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T + np.array(direction) * RELATIVE_SHIFT
    shifts[0, :] = np.array([0.0, 0.0])

    shifts = np.cumsum(shifts, axis=0)

    return [tuple(s) for s in shifts]


class BandMisalignmentTask(EOTask):
    def __init__(
        self,
        feature: FeaturesSpecification,
        interpolation_method: int = INTER_LINEAR,
        border_value: float = 0.0,
    ):
        """Task for simulating band-to-band misalignment

        :param feature: Input feature holding the bands to misalign and the name of the new feature holding the misaligned bands.
        :param interpolation_method: Interpolation method used for band misalignment. Defaults to Linear.
        :param border_value: Constant value use to pad the misaligned bands. Defaults to 0.
        """
        self.input_feature = self.parse_renamed_feature(feature)
        self.interpolation_method = interpolation_method
        self.border_value = border_value

    def execute(self, eopatch: EOPatch) -> EOPatch:
        feat_type, feat_name, new_feat_name = self.input_feature
        eopatch[feat_type][new_feat_name] = np.zeros_like(eopatch[feat_type][feat_name])
        shift_dict = {}

        direction = random.choice(DIRECTIONS)

        for ts_idx, eop_ts in enumerate(eopatch[feat_type][feat_name]):
            warp_matrix = np.eye(3)[:2, :]

            shift_vectors = get_shifts(direction)

            shift_dict[ts_idx] = shift_vectors

            bands_shifted = []

            for b_idx in range(eop_ts.shape[-1]):
                eop_ts_band = eop_ts[..., b_idx]
                warp_matrix[:, 2] = shift_vectors[b_idx]

                band = warpAffine(
                    src=eop_ts_band,
                    M=warp_matrix,
                    dsize=eop_ts_band.T.shape,
                    flags=self.interpolation_method,
                    borderMode=BORDER_CONSTANT,
                    borderValue=self.border_value,
                )

                bands_shifted.append(band[..., np.newaxis])

            eopatch[feat_type][new_feat_name][ts_idx] = np.concatenate(bands_shifted, axis=-1)

        eopatch.meta_info["Shifts"] = shift_dict
        return eopatch


def get_sun_earth_distance(doy: int):
    return 1 - 0.01672 * np.cos(0.9856 * (doy - 4) * np.pi / 180)


class CalculateReflectanceTask(EOTask):
    def __init__(
        self,
        feature: FeaturesSpecification,
        processing_level: ProcessingLevels,
    ):
        """Calculate reflectances from radiances using solar irradiance and distance
        Earth-Sun if L1C level is requested. Otherwise radiance are returned.

        :param feature: Input feature holding radiance values and name of new feature.
        :param processing_level: Processing level desired. If L1A or L1B no conversion to reflectances is applied.
        """
        self.feature = self.parse_renamed_feature(feature)
        self.processing_level = processing_level

    def execute(self, eopatch: EOPatch) -> EOPatch:
        if self.processing_level.value == ProcessingLevels.L1C.value:
            assert len(eopatch.timestamp) == 1, "Implementation supports one timestamp only"

            doy = eopatch.timestamp[0].timetuple().tm_yday
            sun_zenith = eopatch.meta_info["sun_zenith"]
            sun_earth_dist = get_sun_earth_distance(doy)

            reflectances = (
                np.pi
                * sun_earth_dist**2
                * eopatch[self.feature[0]][self.feature[1]]
                / (np.cos(np.radians(sun_zenith)) * HSI_IRR.reshape(1, 1, 1, len(HSI_IRR)))
            )
            eopatch[self.feature[0]][self.feature[2]] = reflectances
        else:
            eopatch[self.feature[0]][self.feature[2]] = eopatch[self.feature[0]][self.feature[1]]
        return eopatch


class GriddingTask(EOTask):
    def __init__(
        self,
        raster_feature: Tuple[FeatureType, str],
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        size: int,
        overlap: float,
        resolution: float,
        time_index: int = 0,
    ):
        """Split the AOI into smaller image chips to create an AI-ready dataset.

        :param raster_feature: A data feature to crop
        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param size: A size of of images to crop out of input image.
        :param overlap: Overlap between sub-images extracted.
        :param resolution: Resolution on which task is running.
        """
        self.raster_feature = raster_feature
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.size = size
        self.overlap = overlap
        self.resolution = resolution
        self.time_index = time_index

    def _grid_data(self, data: np.array) -> Tuple[List[np.array], Dict]:
        height, width, bands = data.shape
        stride = int(self.size * (1 - self.overlap))

        gridded_data = []
        stats = defaultdict(list)
        for x in range(0, width, stride):
            for y in range(0, height, stride):
                x2, y2 = min(x + self.size, width), min(y + self.size, height)
                x1, y1 = max(0, x2 - self.size), max(0, y2 - self.size)

                if x1 == x2 or y1 == y2:
                    continue

                data_slice = data[y1:y2, x1:x2, ...]
                gridded_data.append(data_slice)

                polygon = shapely.geometry.box(x1, y1, x2, y2)
                stats["pixel_geometry"].append(polygon)

        gridded_data = (
            np.stack(gridded_data, axis=0)
            if gridded_data
            else np.zeros((0, self.size, self.size, bands), dtype=data.dtype)
        )
        return gridded_data, stats

    def execute(self, eopatch: EOPatch) -> EOPatch:
        data = eopatch[self.raster_feature][self.time_index]
        gridded_data, stats = self._grid_data(data)

        eopatch[self.data_stack_feature] = gridded_data

        if self.grid_feature:
            crop_grid_gdf = self.calculate_crop_grid(eopatch, stats)
            eopatch[self.grid_feature] = crop_grid_gdf

        return eopatch

    def calculate_crop_grid(self, eopatch: EOPatch, stats: Dict) -> gpd.GeoDataFrame:
        transform = eopatch.bbox.get_transform_vector(self.resolution, self.resolution)

        def pixel_to_utm_transformer(column, row):
            return pixel_to_utm(row, column, transform=transform)

        utm_polygons = [shapely.ops.transform(pixel_to_utm_transformer, polygon) for polygon in stats["pixel_geometry"]]
        crop_grid_gdf = gpd.GeoDataFrame(stats, geometry=utm_polygons, crs=eopatch.bbox.crs.pyproj_crs())
        return crop_grid_gdf


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """Calculate the extent (bounds) of the patch.
    :param eopatch: EOPatch for which the extent is calculated.
    :return: The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    """
    return (
        eopatch.bbox.min_x,
        eopatch.bbox.max_x,
        eopatch.bbox.min_y,
        eopatch.bbox.max_y,
    )


class ExportGridToTiff(EOTask):
    def __init__(
        self,
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        out_folder: str,
        time_index: int = 0,
    ):
        """Export the image chips as GeoTiff files.

        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param out_folder: Path to folder where tiff will be saved.
        :time_index: Timestamp index in EOPatch from which the grid was created.
        """
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.out_folder = out_folder
        self.time_index = time_index

        self.export_task = ExportToTiffTask(
            feature=(FeatureType.DATA, "TEMP_CELL"),
            folder=self.out_folder,
            date_indices=[self.time_index],
        )

    def execute(self, eopatch: EOPatch, *, prefix: Optional[str] = None) -> EOPatch:
        gridded_data = eopatch[self.data_stack_feature]
        gdf = eopatch[self.grid_feature]

        for n_row, row in gdf.iterrows():
            temp_eop = EOPatch(
                data={"TEMP_CELL": gridded_data[[n_row]]},
                bbox=BBox(row.geometry, crs=eopatch.bbox.crs),
                timestamp=[eopatch.timestamp[self.time_index]],
            )

            timestamp_str = eopatch.timestamp[self.time_index].strftime("%Y-%m-%dT%H-%M-%S")
            bbox_str = f"{int(eopatch.bbox.middle[0])}-{int(eopatch.bbox.middle[1])}"
            utm_crs_str = f"{eopatch.bbox.crs.epsg}"

            filename = f"{bbox_str}_{utm_crs_str}_{self.grid_feature[1]}_{timestamp_str}_{n_row:03d}.tiff"
            if prefix is not None:
                filename = f"{prefix}_{self.grid_feature[1]}_{n_row:03d}.tiff"

            self.export_task(
                temp_eop,
                filename=filename,
            )
            del temp_eop

        return eopatch
