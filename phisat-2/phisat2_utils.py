import os
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.ops
from cv2 import warpAffine
from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.io import ExportToTiffTask
from phisat2_constants import (
    BBOX_SIZE_CROPPED,
    CROP_SIZE,
    L1A_RAND_MEAN,
    L1A_RAND_STD,
    L1A_RELATIVE_SHIFTS,
    PAN_WEIGHTS,
    PHISAT2_RESOLUTION,
    S2_BANDS,
    S2_PAN_BANDS,
    WORLD_GDF,
    ProcessingLevels,
)
from sentinelhub import (
    BBox,
    DataCollection,
    SentinelHubCatalog,
    SHConfig,
    parse_time,
    pixel_to_utm,
)
from sentinelhub.aws.request import AwsProductRequest


class SCLCloudTask(EOTask):
    def __init__(self, scl_feature: Tuple[FeatureType, str]):
        """Extract cloud-related info from the provided SCL layer in separate features and deletes the SCL feature.

        :param scl_feature: Name of feature in EOPatch holding the scene classification mask.
        """
        self.scl_feature = self.parse_feature(scl_feature)
        self.scl_cloud_feature = (FeatureType.MASK, "SCL_CLOUD")
        self.scl_cloud_shadow_feature = (FeatureType.MASK, "SCL_CLOUD_SHADOW")
        self.scl_cirrus_feature = (FeatureType.MASK, "SCL_CIRRUS")

    def execute(self, eopatch: EOPatch) -> EOPatch:
        scl = eopatch[self.scl_feature]
        eopatch[self.scl_cloud_feature] = ((scl == 8) | (scl == 9)).astype(np.uint8)
        eopatch[self.scl_cloud_shadow_feature] = (scl == 3).astype(np.uint8)
        eopatch[self.scl_cirrus_feature] = (scl == 10).astype(np.uint8)
        del eopatch[self.scl_feature]

        return eopatch


class AddPANBandTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
    ):
        """Calculate pan-chromatic band as weighted average of other bands.

        :param input_feature: Input feature holding the Sentinel-2 bands used to calculate the pan-chromatic band.
        :param output_feature: Output feature holding Sentinel-2 bands and pan-chromatic band.
            The pan-chromatic band is inserted at index 3.
        """
        self.input_feature = self.parse_feature(
            input_feature, allowed_feature_types=[FeatureType.DATA]
        )
        self.output_feature = self.parse_feature(
            output_feature, allowed_feature_types=[FeatureType.DATA]
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bands = eopatch[self.input_feature]

        assert (
            len(PAN_WEIGHTS) == bands.shape[-1]
        ), "The number of bands of the input features must be 7"

        pan_band = np.sum(bands * np.array(PAN_WEIGHTS) / sum(PAN_WEIGHTS), axis=-1)

        pan_index = S2_PAN_BANDS.index("PAN")
        new_bands = np.insert(bands, pan_index, pan_band, axis=-1)

        eopatch[self.output_feature] = new_bands

        return eopatch


class AddMetadataTask(EOTask):
    def __init__(self, config: Optional[SHConfig] = None):
        """Download Sentinel-2 metadata necessary to compute radiances

        :param config: Optional Sentinel Hub configuration file.
        """
        self.config = config

    @staticmethod
    def filter_and_sort_tiles(
        tiles: List[Dict], timestamps: List[datetime]
    ) -> List[Dict]:
        filtered_tiles = []
        available_dates = []
        for tile in tiles:
            tile_dt = parse_time(tile["properties"]["datetime"], ignoretz=True)
            if (tile_dt in timestamps) and not (tile_dt.date() in available_dates):
                tile["timestamp"] = tile_dt
                available_dates.append(tile_dt.date())
                filtered_tiles.append(tile)
        if len(filtered_tiles) != len(timestamps):
            raise ValueError(
                f"Expected {len(timestamps)} tiles, got {len(filtered_tiles)}!"
            )
        return sorted(filtered_tiles, key=lambda item: item["timestamp"])

    def execute(self, eopatch: EOPatch, **kwargs) -> EOPatch:
        if not all([eopatch, eopatch.bbox, eopatch.timestamp]):
            raise ValueError(
                "AddMetadataTask needs eopatch to have bbox and temporal data!"
            )

        # the metadata info location has been changed since 2022-01-25
        catalog = SentinelHubCatalog(self.config)
        query = catalog.search(
            collection=DataCollection.SENTINEL2_L2A,
            bbox=eopatch.bbox,
            time=[eopatch.timestamp[0], eopatch.timestamp[-1]],
        )
        tiles = self.filter_and_sort_tiles(list(query), eopatch.timestamp)

        dim = len(eopatch.timestamp)
        eopatch.scalar["earth_sun_dist"] = np.ones((dim, 1))
        for s2_band in S2_BANDS:
            eopatch.scalar[f"sol_irr_{s2_band}"] = np.ones((dim, 1))

        for tile_idx, (tile, timestamp) in enumerate(zip(tiles, eopatch.timestamp)):
            metadata = AwsProductRequest(
                product_id=tile["id"], bands=[], metafiles=["metadata"]
            ).get_data()

            index_refl_conversion = 4
            if timestamp >= datetime.strptime("2022-01-25", "%Y-%m-%d"):
                index_refl_conversion = 5

            assert (
                metadata[0][0][1][index_refl_conversion][0].tag == "U"
            ), "Issue indexing metadata"
            assert (
                metadata[0][0][1][index_refl_conversion][1].tag
                == "Solar_Irradiance_List"
            ), "Issue indexing metadata"

            eopatch.scalar["earth_sun_dist"][tile_idx] = float(
                metadata[0][0][1][index_refl_conversion][0].text
            )

            solar_irradiance_list = metadata[0][0][1][index_refl_conversion][1]
            for s2_idx, s2_band in enumerate(DataCollection.SENTINEL2_L1C.bands):
                if s2_band.name in S2_BANDS:
                    eopatch.scalar[f"sol_irr_{s2_band.name}"][tile_idx] = float(
                        solar_irradiance_list[s2_idx].text
                    )

        return eopatch


def get_shifts_l1a() -> List[Tuple[float, float]]:
    """Compute random shifts for L1A level"""

    mis_amplitude = np.random.normal(
        L1A_RAND_MEAN, L1A_RAND_STD, size=(len(S2_PAN_BANDS),)
    ) + np.array(L1A_RELATIVE_SHIFTS)
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T

    shifts[0, :] = np.array([0.0, 0.0])
    shifts = np.flip(np.cumsum(shifts, axis=0), axis=0)

    return [tuple(s) for s in shifts]


def get_shifts_l1b(rand_std: int) -> List[Tuple[float, float]]:
    """Compute random shifts for L1B level"""

    mis_amplitude = np.random.normal(0, rand_std, size=(len(S2_PAN_BANDS),))
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(S2_PAN_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T
    shifts[2, :] = np.array([0.0, 0.0])

    return [tuple(s) for s in shifts.tolist()]


class BandMisalignmentTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        processing_level: ProcessingLevels,
        std_sea: int = 6,
        interpolation_method: int = cv2.INTER_LINEAR,
    ):
        """Task for simulating L1A or L1B band misalignment

        :param input_feature: Input feature holding the bands to misalign.
        :param output_feature: Output feature with misaligned bands according to processing level.
        :param processing_level: Processing level that defines which band misalignment method is applied.
        :param std_sea: Standard deviation for AOIs over sea. Defaults to 6.
        :param interpolation_method: Interpolation method used in misalignment. Defaults to INTER_LINEAR.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)
        self.processing_level = processing_level
        self.std_sea = std_sea
        self.interpolation_method = interpolation_method

    def execute(self, eopatch: EOPatch) -> EOPatch:
        eopatch[self.output_feature] = eopatch[self.input_feature].copy()
        shift_dict = {}

        patch_geom = eopatch.bbox.transform(WORLD_GDF.crs.to_epsg()).geometry
        is_in_water = not WORLD_GDF.intersects(patch_geom).any()
        rand_std = self.std_sea if is_in_water else 1

        for ts_idx, eop_ts in enumerate(eopatch[self.input_feature]):

            warp_matrix = np.eye(3)[:2, :]

            if self.processing_level.value == ProcessingLevels.L1A.value:
                shift_vectors = get_shifts_l1a()
            else:
                shift_vectors = get_shifts_l1b(rand_std)

            shift_dict[ts_idx] = shift_vectors

            bands_shifted = []

            for b_idx in range(eop_ts.shape[-1]):

                eop_ts_band = eop_ts[..., b_idx]
                warp_matrix[:, 2] = shift_vectors[b_idx]

                band = warpAffine(
                    src=eop_ts_band,
                    M=warp_matrix,
                    dsize=eop_ts_band.shape,
                    flags=self.interpolation_method,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                bands_shifted.append(band)

            eopatch[self.output_feature][ts_idx] = np.moveaxis(
                np.array(bands_shifted), 0, -1
            )

        eopatch.meta_info["Shifts"] = shift_dict
        return eopatch


class CropTask(EOTask):
    def __init__(self, features_to_crop: List[Tuple[FeatureType, str]]):
        """Remove pixels at the boundary that contain misalignment artefacts

        :param features_to_crop: List o features that will be cropped in-place.
        """
        self.features_to_crop = self.parse_features(features_to_crop)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bbox_size = eopatch.bbox.upper_right[0] - eopatch.bbox.lower_left[0]

        if bbox_size < BBOX_SIZE_CROPPED:
            return eopatch

        crop_h, crop_w = CROP_SIZE
        for feature in self.features_to_crop:
            eopatch[feature] = eopatch[feature][:, crop_h:-crop_h, crop_w:-crop_w, :]

        eopatch.bbox = eopatch.bbox.buffer(
            (-crop_h * PHISAT2_RESOLUTION, -crop_w * PHISAT2_RESOLUTION),
            relative=False,
        )

        return eopatch


class CalculateRadianceTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
    ):
        """Calculate radiances from reflectances using solar irradiance and distance Earth-Sun.

        :param input_feature: Input feature holding reflectance values. Expected 7 bands.
        :param output_feature: Output feature with radiance values.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)

    def execute(self, eopatch):
        assert all(
            [
                isinstance(eopatch.scalar[f"sol_irr_{band}"], np.ndarray)
                for band in S2_BANDS
            ]
        )
        assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

        factor = (
            np.cos(np.radians(eopatch.data["sunZenithAngles"]))
            * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
            / np.pi
        )
        solar_irradiances = np.concatenate(
            [eopatch.scalar[f"sol_irr_{band}"][:, :, np.newaxis] for band in S2_BANDS],
            axis=-1,
        )

        radiances = (
            eopatch[self.input_feature]
            * factor
            * solar_irradiances[:, :, np.newaxis, :]
        )
        eopatch[self.output_feature] = radiances
        return eopatch


class CalculateReflectanceTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        processing_level: ProcessingLevels,
    ):
        """Calculate reflectances from radiances using solar irradiance and distance
        Earth-Sun if L1C level is requested. Otherwise radiance are returned.

        :param input_feature: Input feature holding radiance values. Expected 8 bands.
        :param output_feature: Output feature with radiances (for L1A and L1B) or reflectance (for L1C) values.
        :param processing_level: Processing level desired. If L1A or L1B no conversion to reflectances is applied.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)
        self.processing_level = processing_level

    def execute(self, eopatch: EOPatch) -> EOPatch:
        if self.processing_level.value == ProcessingLevels.L1C.value:
            assert all(
                [
                    isinstance(eopatch.scalar[f"sol_irr_{band}"], np.ndarray)
                    for band in S2_BANDS
                ]
            )
            assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

            # We don't have the irradiance for the PAN band, so we compute it as weighted sum
            sol_irr_pan = 0.0
            for nband, band in enumerate(S2_BANDS):
                sol_irr_pan += eopatch.scalar[f"sol_irr_{band}"] * PAN_WEIGHTS[nband]

            eopatch.scalar["sol_irr_PAN"] = sol_irr_pan

            factor = (
                np.cos(np.radians(eopatch.data["sunZenithAngles_RES"]))
                * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
                / np.pi
            )
            solar_irradiances = np.concatenate(
                [
                    eopatch.scalar[f"sol_irr_{band}"][:, :, np.newaxis]
                    for band in S2_PAN_BANDS
                ],
                axis=-1,
            )

            reflectances = eopatch[self.input_feature] / (
                factor * solar_irradiances[:, :, np.newaxis, :]
            )
            eopatch[self.output_feature] = reflectances
        else:
            eopatch[self.output_feature] = eopatch[self.input_feature]
        return eopatch


class PhisatCalculationTask(EOTask):
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
        :param calculation: Which simulation to execute, i.e. "SNR" or "PSF"
        """
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.executable = executable
        self.calculation = calculation

    def execute(self, eopatch: EOPatch) -> EOPatch:
        with tempfile.TemporaryDirectory(
            prefix="phisat2_calc", suffix=self.calculation
        ) as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")

            # temporary write input feature to numpy
            np.save(input_npy, eopatch[self.input_feature])

            # run exec
            subprocess.run(
                f"{self.executable} {self.calculation} {input_npy} {output_npy}".split(
                    " "
                ),
                check=True,
            )

            # read temp output
            eopatch[self.output_feature] = np.load(output_npy)

        # return eopatch with new feature
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

        utm_polygons = [
            shapely.ops.transform(pixel_to_utm_transformer, polygon)
            for polygon in stats["pixel_geometry"]
        ]
        crop_grid_gdf = gpd.GeoDataFrame(
            stats, geometry=utm_polygons, crs=eopatch.bbox.crs.pyproj_crs()
        )
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

    def execute(self, eopatch: EOPatch) -> EOPatch:

        gridded_data = eopatch[self.data_stack_feature]
        gdf = eopatch[self.grid_feature]

        for n_row, row in gdf.iterrows():

            temp_eop = EOPatch(
                data={"TEMP_CELL": gridded_data[[n_row]]},
                bbox=BBox(row.geometry, crs=eopatch.bbox.crs),
                timestamp=[eopatch.timestamp[self.time_index]],
            )

            timestamp_str = eopatch.timestamp[self.time_index].strftime(
                "%Y-%m-%dT%H-%M-%S"
            )
            bbox_str = f"{int(eopatch.bbox.middle[0])}-{int(eopatch.bbox.middle[1])}"
            utm_crs_str = f"{eopatch.bbox.crs.epsg}"
            self.export_task(
                temp_eop,
                filename=f"{bbox_str}_{utm_crs_str}_{self.grid_feature[1]}_{timestamp_str}_{n_row:03d}.tiff",
            )
            del temp_eop

        return eopatch
