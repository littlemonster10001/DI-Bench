from .image_retrieval import ImageRetrievalPrompter
from .cross_view_matching import ImageLevelCrossViewMatchingPrompter, ObjectLevelCrossViewMatchingPrompter, ObjectLevelCrossViewMatchingTextPrompter
from .damage_assessment import BuildingDamageAssessmentPrompter, BuildingDamageAssessmentTextPrompter, BuildingDamageCountingPrompter, BuildingDamageCountingTextPrompter, RoadDamageReasoningPrompter
from .alignment_tasks import POIAlignmentPrompter, POIAlignmentTextPrompter, PopulationEstimationPrompter, PopulationEstimationTextPrompter, HeightAlignmentPrompter, HeightComparisonPrompter, HeightComparisonTextPrompter
from .measurement_tasks import AreaEstimationPrompter, AreaEstimationTextPrompter, LengthEstimationPrompter, LengthEstimationTextPrompter, DistanceEstimationPrompter, DistanceEstimationTextPrompter
from .routing_and_uav import RoutePlanningPrompter, RoutePlanningTextPrompter, UAVLandingAssessmentPrompter, UAVLandingAssessmentTextPrompter

# V2 版本专属的路由注册表
PROMPTER_REGISTRY_V2 = {
    'Image_Retrieval': ImageRetrievalPrompter(),
    'Image_Level_Cross_View_Matching': ImageLevelCrossViewMatchingPrompter(),
    'Object_Level_Cross_View_Matching': ObjectLevelCrossViewMatchingPrompter(),
    'Object_Level_Cross_View_Matching_Text': ObjectLevelCrossViewMatchingTextPrompter(),
    'building_damage_assessment': BuildingDamageAssessmentPrompter(),
    'building_damage_assessment_Text': BuildingDamageAssessmentTextPrompter(),
    'Building_Damage_Counting': BuildingDamageCountingPrompter(),
    'Building_Damage_Counting_Text': BuildingDamageCountingTextPrompter(),
    'Road_Damage_Reasoning': RoadDamageReasoningPrompter(),
    'poi_alignment': POIAlignmentPrompter(),
    'poi_alignment_Text': POIAlignmentTextPrompter(),
    'population_estimation': PopulationEstimationPrompter(),
    'population_estimation_Text': PopulationEstimationTextPrompter(),
    'height_alignment': HeightAlignmentPrompter(),
    'height_comparison': HeightComparisonPrompter(),
    'height_comparison_Text': HeightComparisonTextPrompter(),
    'Area_Estimation': AreaEstimationPrompter(),
    'Area_Estimation_Text': AreaEstimationTextPrompter(),
    'Length_Estimation': LengthEstimationPrompter(),
    'Length_Estimation_Text': LengthEstimationTextPrompter(),
    'Distance_Estimation': DistanceEstimationPrompter(),
    'Distance_Estimation_Text': DistanceEstimationTextPrompter(),
    'route_planning': RoutePlanningPrompter(),
    'route_planning_Text': RoutePlanningTextPrompter(),
    'uav_landing_assessment': UAVLandingAssessmentPrompter(),
    'uav_landing_assessment_Text': UAVLandingAssessmentTextPrompter(),
}

def get_prompter_v2(task_type: str):
    if task_type not in PROMPTER_REGISTRY_V2:
        raise ValueError(f"Unknown V2 task type: {task_type}")
    return PROMPTER_REGISTRY_V2[task_type]


def get_prompter(task_type: str):
    return get_prompter_v2(task_type)
