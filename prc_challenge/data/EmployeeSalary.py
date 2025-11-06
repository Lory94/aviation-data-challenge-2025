from .SupervisedDataset import SupervisedDataset
from skrub.datasets import fetch_employee_salaries
from sklearn.model_selection import train_test_split


class EmployeeSalary(SupervisedDataset):

    def __init__(self):

        dataset = fetch_employee_salaries()
        employees_df, salaries = dataset.X, dataset.y

        train_X, test_X, train_Y, test_Y = train_test_split(employees_df, salaries, test_size=200)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            test_X=test_X,
            test_Y=test_Y,
            features={	
                "nominal_category": ["gender", "department", "department_name", "division", "assignment_category", "employee_position_title"],
                "ordinal_category": [],
                "interval_numeric": ["year_first_hired"],  # No absolute zero
                "ratio_numeric": [],  # Absolute zero exists
                "natural_text": [],
                "date": ["date_first_hired"],
            }
        )