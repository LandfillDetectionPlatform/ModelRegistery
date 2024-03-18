
from zenml import get_step_context, step
from zenml.client import Client
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def model_promoter(metrics: dict, precision_threshold: float, recall_threshold: float) -> bool:
    
    precision = metrics.get("precision", 0)
    recall = metrics.get("recall", 0)

    is_promoted = False

    if precision <= precision_threshold and recall <= recall_threshold:
        logger.info(
            f"Model precision {precision*100:.2f}% and Model recall {recall*100:.2f} %is are bellow 80% ! Not promoting model."
        )
        print('rejected')
    else:
        logger.info(f"Model promoted!")
        print('accepted')

        is_promoted = True

    return is_promoted